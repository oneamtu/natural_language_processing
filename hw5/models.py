import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import random
from torch.autograd import Variable as Var
from utils import *
from data import *
from lf_evaluator import *

import pickle
import numpy as np
from typing import List

# catch NaNs
torch.autograd.set_detect_anomaly(True)

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=20, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    # NOTE - 200 works better for embedding size
    parser.add_argument('--embedding_size', type=int, default=200, help='embedding size (number of dimensions)')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden layer size (number of dimensions)')


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


###################################################################################################################
# You do not have to use any of the classes in this file, but they're meant to give you a starting implementation.
# for your network.
###################################################################################################################

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, embedding_dropout=0.2, bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer

        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)

        self.output_emb = EmbeddingLayer(emb_dim, len(output_indexer), embedding_dropout)
        self.decoder = nn.LSTMCell(emb_dim, hidden_size)

        self.attention_layer = AttentionLayer(hidden_size)

        self.output_size = len(output_indexer)
        self.hidden_to_out = nn.Linear(hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()

    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, teacher_forcing_rate=1.0):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len] vector of indices or a batched input/output
        [batch size x sent len]. y_tensor contains the gold sequence(s) used for training
        :param inp_lens_tensor/out_lens_tensor: either a vector of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        (enc_output_each_word, enc_context_mask, h_t) = self.encode_input(x_tensor, inp_lens_tensor)
        # initialize decoder state from encoder state
        attention_h_t, c_t = h_t

        total_loss = 0

        # init start symbol for batch
        decoder_input = torch.ones((y_tensor.size(0)), dtype=torch.long) * self.output_indexer.index_of(SOS_SYMBOL)
        decoder_input_emb = self.output_emb.forward(decoder_input)

        torch.transpose(y_tensor, 0, 1)
        teacher_forcing = random.random() < teacher_forcing_rate

        for word_i in range(out_lens_tensor.max()):
            h_t, c_t = self.decoder(decoder_input_emb, (attention_h_t, c_t))

            # if any([torch.isnan(x) for x in h_t.view(-1)]):
            #     import ipdb; ipdb.set_trace()

            attention_h_t = self.attention_layer(enc_output_each_word, h_t, enc_context_mask)
            # if any([torch.isnan(x) for x in attention_h_t.view(-1)]):
            #     import ipdb; ipdb.set_trace()

            output = self.hidden_to_out(attention_h_t)
            log_probs = self.softmax(output)

            loss = self.nll_loss(log_probs, y_tensor[:, word_i])
            # TODO: ignore loss for 'out of length' seqs?
            # see ignore_index option
            # or use context_mask
            total_loss += loss

            if teacher_forcing:
                decoder_input = y_tensor[:, word_i]
            else:
                decoder_input = torch.argmax(log_probs, dim=1)

            decoder_input_emb = self.output_emb.forward(decoder_input)

        return total_loss

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        derivations = []

        for ex in test_data:
            derivation = []

            # batch of 1
            x_tensor = torch.tensor(ex.x_indexed).view(1, -1)

            (enc_output_each_word, enc_context_mask, h_t) = self.encode_input(x_tensor, torch.tensor([len(ex.x_indexed)]))
            attention_h_t, c_t = h_t

            decoder_input = torch.ones((1), dtype=torch.long) * self.output_indexer.index_of(SOS_SYMBOL)
            decoder_input_emb = self.output_emb.forward(decoder_input)

            word_index = -1

            # init start symbol for batch
            # import ipdb; ipdb.set_trace()
            while word_index != self.output_indexer.index_of(EOS_SYMBOL) and \
                len(derivation) <= 100:
                h_t, c_t = self.decoder(decoder_input_emb, (attention_h_t, c_t))
                attention_h_t = self.attention_layer(enc_output_each_word, h_t, enc_context_mask)
                output = self.hidden_to_out(attention_h_t)
                log_probs = self.softmax(output)

                # import ipdb; ipdb.set_trace()
                decoder_input = torch.argmax(log_probs, dim=1)
                decoder_input_emb = self.output_emb.forward(decoder_input)

                word_index = decoder_input.item()

                if word_index != self.output_indexer.index_of(EOS_SYMBOL):
                    derivation.append(self.output_indexer.get_object(word_index))

                decoder_input = word_index

            derivations.append([Derivation(ex, 1.0, derivation)])

        return derivations

    def encode_input(self, x_tensor, inp_lens_tensor):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple). ONLY THE ENCODER FINAL
        STATES are needed for the basic seq2seq model. enc_output_each_word is needed for attention, and
        enc_context_mask is needed to batch attention.

        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0, 0],
        [1, 2, 3, 0],
        [2, 0, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = self.input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states) = self.encoder.forward(input_emb, inp_lens_tensor)
        return (enc_output_each_word, enc_context_mask, enc_final_states)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_emb_dim: int, hidden_size: int, bidirect: bool):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_emb_dim, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        Note that output is only needed for attention, and context_mask is only used for batched attention.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = max(input_lens.data).item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            new_output = self.reduce_h_W(output)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (new_output, context_mask, h_t)

class AttentionLayer(nn.Module):
    """
    Attention Layer
    Computes the context state based on the encoder hidden states
    and the current hidden state
    """
    def __init__(self, hidden_size: int):
        """
        :param hidden_size: dimensionality of the hidden state
        """
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_scorer = AttentionScorer(hidden_size, 'general')
        self.reduce_attention_context_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, encoder_hidden_states, h_t, encoder_context_mask):
        """
        :param encoder_hidden_states: hidden states computed by the encoder
        :param h_t: current hidden state
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        encoder_h_t_scores = self.attention_scorer.forward(encoder_hidden_states, h_t)
        # import ipdb; ipdb.set_trace()
        log_mask = torch.log(encoder_context_mask.float()).transpose(0, 1)
        encoder_h_t_scores += log_mask
        # use logsumexp to prevent inf (ratio is more stable this way)
        log_sum_exp_encoder_h_t_scores = torch.logsumexp(encoder_h_t_scores, dim=0)
        a_t = torch.exp(encoder_h_t_scores - log_sum_exp_encoder_h_t_scores)

        # einstein notation to do the weighted sum of decoder states
        # and preserve each batch
        c_t = torch.einsum('sb,sbi->bi', a_t, encoder_hidden_states)

        return torch.tanh(self.reduce_attention_context_W(torch.cat((c_t, h_t), dim=1)))

class AttentionScorer(nn.Module):
    """
    Attention score between an input hidden_state and the current
    target state; used for computing attention weights
    """
    def __init__(self, hidden_size: int, scorer_strategy: str):
        """
        :param hidden_size: dimensionality of the hidden state
        :param scorer_strategy: strategies for determining attention score
        Outlined in Luong NMT '15 paper - ('dot', 'general', 'concat')
        """
        super(AttentionScorer, self).__init__()
        self.hidden_size = hidden_size
        self.scorer_strategy = scorer_strategy
        if scorer_strategy == 'general':
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, h_s, h_t):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        if self.scorer_strategy == 'dot':
            # torch doesn't have great support for batched dot product
            # so we compute the full product, and take the diagonal
            # to get the right batched dots h_s[0][0] * h_t[0], etc.
            full_dot = torch.tensordot(h_s, h_t, dims=([2], [1]))
            return torch.diagonal(full_dot, dim1=1, dim2=2)
        elif self.scorer_strategy == 'general':
            # torch doesn't have great support for batched dot product
            # so we compute the full product, and take the diagonal
            # to get the right batched dots h_s[0][0] * h_t[0], etc.
            full_dot = torch.tensordot(self.W_a(h_s), h_t, dims=([2], [1]))
            return torch.diagonal(full_dot, dim1=1, dim2=2)
        else:
            raise Exception(f"Strategy {self.scorer_strategy} no supported!")

###################################################################################################################
# End optional classes
###################################################################################################################


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


TEACHER_FORCING_BASE=1.0
TEACHER_FORCING_EPOCHS=4

def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # train_data = train_data[0:1]
    # dev_data = train_data
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    # all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    # all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # if os.path.exists("parser_2.pickle"):
    #     print("Loading a PICKLE")
    #     seq2seq_parser = pickle.load(open("parser_2.pickle", "rb"))
    # else:
    seq2seq_parser = Seq2SeqSemanticParser(
            input_indexer, output_indexer,
            args.embedding_size, args.hidden_size)

    optimizer = optim.Adam(seq2seq_parser.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        # enable train mode
        seq2seq_parser.train()

        total_loss = 0.0

        ex_indices = [i for i in range(0, len(train_data))]
        random.shuffle(ex_indices)

        for range_start in np.arange(0, len(ex_indices), args.batch_size):
            range_end = min(range_start + args.batch_size, len(ex_indices) - 1)
            if range_end == range_start:
                continue

            sliced_train_exs = ex_indices[range_start:range_end]
            x_tensor = torch.from_numpy(all_train_input_data[sliced_train_exs])
            x_input_lengths = [len(train_data[idx].x_indexed) for idx in sliced_train_exs]
            x_input_lengths_tensor = torch.tensor(x_input_lengths)
            y_tensor = torch.from_numpy(all_train_output_data[sliced_train_exs])
            y_input_lengths = [len(train_data[idx].y_indexed) for idx in sliced_train_exs]
            y_input_lengths_tensor = torch.tensor(y_input_lengths)
            seq2seq_parser.zero_grad()

            loss = seq2seq_parser.forward(
                    x_tensor, x_input_lengths_tensor,
                    y_tensor, y_input_lengths_tensor,
                    teacher_forcing_rate=TEACHER_FORCING_BASE ** max(epoch-TEACHER_FORCING_EPOCHS, 0))
            loss.backward()
            optimizer.step()

            total_loss += loss

        print("Total loss on epoch %i: %f" % (epoch, total_loss))

        # enable eval mode
        seq2seq_parser.eval()

        # Evaluate on the dev set
        if True:
            # import ipdb; ipdb.set_trace()
            evaluate(dev_data, seq2seq_parser, use_java=False)
            pickle.dump(seq2seq_parser, open("parser.pickle", "wb"))

    return seq2seq_parser
