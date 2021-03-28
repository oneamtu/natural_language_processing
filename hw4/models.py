# models.py

import numpy as np
import collections
import random

import torch
import torch.nn as nn
from torch import optim

import pickle
import sys
from typing import List

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

CLASSIFIER_EMBEDDING_SIZE=8
CLASSIFIER_HIDDEN_SIZE=8
CLASSIFIER_OUTPUT_SIZE=2

class RNNClassifier(ConsonantVowelClassifier):

    def __init__(self, vocab_indexer):
        self.vocab_indexer = vocab_indexer
        self.rnn = LMClassifierRNN(CLASSIFIER_EMBEDDING_SIZE, CLASSIFIER_HIDDEN_SIZE, len(vocab_indexer), CLASSIFIER_OUTPUT_SIZE)
        self.optimizer = optim.Adam(self.rnn.parameters())

    def preprocess_input(self, contexts: List[str]) -> torch.Tensor:
        """
        get embedding indexes for each word
        """
        indexed_contexts = []
        for context in contexts:
            char_indexes = [self.vocab_indexer.index_of(c) for c in context]
            indexed_contexts.append(torch.tensor(char_indexes))

        return torch.stack(indexed_contexts)

    def predict(self, context) -> int:
        """
        @see SentimentClassifier#predict
        """
        log_probs = self.rnn.forward(self.preprocess_input([context]))
        # from IPython import embed; embed()
        return torch.argmax(log_probs, dim=1)

    def predict_all(self, contexts: List[str]) -> List[int]:
        """
        @see SentimentClassifier#predict_all
        """
        log_probs = self.rnn.forward(self.preprocess_input(contexts))
        return torch.argmax(log_probs, dim=1)

    def train(self, contexts: List[str], labels: List[int]) -> float:
        """
        NN train step
        NOTE: inspired from FFNN example

        returns training loss
        """
        x = self.preprocess_input(contexts)
        # Zero out the gradients from the torch.from_numpy(np.array(labels, dtype=np.int64))FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
        self.rnn.zero_grad()
        log_probs = self.rnn.forward(x)
        # from IPython import embed; embed()
        loss = nn.NLLLoss()(log_probs, torch.tensor(labels))
        # Computes the gradient and takes the optimizer step
        loss.backward()
        self.optimizer.step()

        return loss

class LMClassifierRNN(nn.Module):
    """
    Defines the core neural network for doing sentiment classification
    over a single datapoint at a time.

    It consists of an embedding average processing layer
    and multiple hidden layers.

    The forward() function does the important computation.
    The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, embedding_size, hidden_size, vocab_size, out_size):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param embedding_size: size of embedding (integer)
        :param hidden_size: size of hidden layer (integer)
        :param vocab_size: size of vocab (integer)
        :param out_size: size of output (integer), which should be the number of classes - 2 for vowels, consonants
        """
        super(LMClassifierRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

        self.hidden_to_out = nn.Linear(hidden_size, out_size)
        self.output_size = hidden_size

        # Initialize weights according to a formula due to Xavier Glorot.
        self.init_weights()

    """
    Initialize the LSTM weights
    Inspired from lstm_lecture.py, using Glorot
    """
    def init_weights(self):
        # This is a randomly initialized RNN.
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)

    def forward(self, x: torch.Tensor):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: an array of character indices
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        # print(x.size())
        embedded = self.embedding(x)
        # print(embedded.size())
        lstm_output, (hidden_state, cell_state) = self.lstm(embedded)
        # print(hidden_state.size())

        # hidden_state has length 1 instead of seq length, so drop it
        output = self.hidden_to_out(hidden_state.view(x.size()[0], self.output_size))
        # import ipdb; ipdb.set_trace()

        # print(output.size())
        return nn.LogSoftmax(dim=1)(output)

CLASSIFIER_NUM_EPOCHS=18
CLASSIFIER_BATCH_SIZE=10

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    classifier = RNNClassifier(vocab_index)

    # train_cons_exs = train_cons_exs[0:100]
    # dev_cons_exs = train_cons_exs[0:100]
    # train_vowel_exs = train_vowel_exs[0:100]
    # dev_vowel_exs = train_vowel_exs[0:100]

    for epoch in range(CLASSIFIER_NUM_EPOCHS):
        classifier.rnn.train()

        ex_indices = [i for i in range(0, len(train_vowel_exs) + len(train_cons_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for range_start in np.arange(0, len(ex_indices), CLASSIFIER_BATCH_SIZE):
            range_end = min(range_start + CLASSIFIER_BATCH_SIZE, len(ex_indices) - 1)
            if range_end == range_start:
                continue

            sliced_train_exs = []
            labels = []

            for i in ex_indices[range_start:range_end]:
                if i < len(train_vowel_exs):
                    sliced_train_exs.append(train_vowel_exs[i])
                    # vowel gold - 1
                    labels.append(1)
                else:
                    sliced_train_exs.append(train_cons_exs[i % len(train_vowel_exs)])
                    # cons gold - 0
                    labels.append(0)

            loss = classifier.train(sliced_train_exs, labels)
            total_loss += loss

        print("Total loss on epoch %i: %f" % (epoch, total_loss))

        classifier.rnn.eval()
        evaluate = True

        # Evaluate on the dev set
        if evaluate:
            num_correct = 0
            num_total = 0

            # import ipdb; ipdb.set_trace()

            predictions = classifier.predict_all(dev_vowel_exs)
            for idx in range(0, len(dev_vowel_exs)):
                if 1 == predictions[idx]:
                    num_correct += 1
                num_total += 1

            predictions = classifier.predict_all(dev_cons_exs)
            for idx in range(0, len(dev_cons_exs)):
                if 0 == predictions[idx]:
                    num_correct += 1
                num_total += 1

            # from IPython import embed; embed()
            acc = float(num_correct) / num_total
            print("Epoch %i dev - Accuracy: %i / %i = %f" % (epoch, num_correct, num_total, acc))
            pickle.dump(classifier, open("classifier.pickle", "wb"))

    return classifier


if __name__=="__main__":
    classifier = pickle.load(open("classifier.pickle", "rb"))
    sentence = ' '.join(sys.argv[1:])
    prediction = classifier.predict(sentence)
    print(f"Prediction is: {prediction}")

    log_probs = classifier.rnn.forward(classifier.preprocess_input([sentence]))
    # from IPython import embed; embed()
    # import ipdb; ipdb.set_trace()
    assert(torch.exp(log_probs).sum() == 1.)
    print(f"Probs are: {torch.exp(log_probs)}")

#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

# TODO: update?
LM_EMBEDDING_SIZE=16
LM_HIDDEN_SIZE=16

class RNNLanguageModel(LanguageModel):
    def __init__(self, vocab_indexer):
        self.vocab_indexer = vocab_indexer
        self.rnn = LMRNN(LM_EMBEDDING_SIZE, LM_HIDDEN_SIZE, len(vocab_indexer), len(vocab_indexer))
        # TODO: lr=0.001
        self.optimizer = optim.Adam(self.rnn.parameters())

    def preprocess_input(self, contexts: List[str]) -> torch.Tensor:
        """
        get embedding indexes for each word
        """
        indexed_contexts = []
        for context in contexts:
            char_indexes = [self.vocab_indexer.index_of(c) for c in context]
            indexed_contexts.append(torch.tensor(char_indexes))

        return torch.stack(indexed_contexts)

    def get_next_char_log_probs(self, context):
        _, hidden_state, _ = self.rnn.forward(self.preprocess_input([" " + context]))
        # discard batch dim == size 1 (similar to unsqueeze)
        output = self.rnn.hidden_to_out(hidden_state.view(-1, LM_HIDDEN_SIZE))
        log_probs = nn.LogSoftmax(dim=1)(output)
        return log_probs.view(len(self.vocab_indexer)).detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        # 'burn in' the context to get probs up to the last
        # character, which we discard, since we only care about
        # probs on the next_chars; we'll just use the states in
        # the next pass
        _, hidden_state, cell_state = self.rnn.forward(
                self.preprocess_input([" " + context[0:-1]]))
        # the last char of the context provides the probs for the first
        # character in next_chars, so add it in; drop the last char
        # since we don't need to 'predict' what comes after it
        seq_hidden, _, _ = self.rnn.forward(
                self.preprocess_input([context[-1] + next_chars[0:-1]]),
                prev_hidden_state=hidden_state,
                prev_cell_state=cell_state)

        # import ipdb; ipdb.set_trace()
        # discard batch dim == size 1
        seq_output = self.rnn.hidden_to_out(seq_hidden).view(len(next_chars), len(self.vocab_indexer))
        seq_log_probs = nn.LogSoftmax(dim=1)(seq_output)
        # print(seq_log_probs.size())
        return sum([seq_log_probs[i, self.vocab_indexer.index_of(c)].detach().numpy() for i, c in enumerate(next_chars)])

    def train(self, contexts: List[str], labels: List[List[int]],
            prev_hidden_state: torch.tensor, prev_cell_state: torch.tensor) -> float:
        """
        NN train step
        NOTE: inspired from FFNN example

        returns training loss
        """
        x = self.preprocess_input(contexts)
        # Zero out the gradients from the torch.from_numpy(np.array(labels, dtype=np.int64))FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
        self.rnn.zero_grad()
        seq_hidden, hidden_state, cell_state = self.rnn.forward(x, prev_hidden_state, prev_cell_state)
        seq_output = self.rnn.hidden_to_out(seq_hidden)
        seq_log_probs = nn.LogSoftmax(dim=2)(seq_output)

        # import ipdb; ipdb.set_trace()
        total_loss = 0
        for i in range(len(seq_log_probs)):
            total_loss += nn.NLLLoss()(seq_log_probs[i], torch.tensor(labels[i]))

        # Computes the gradient and takes the optimizer step
        total_loss.backward()
        self.optimizer.step()

        # detach from computation graph, only values are needed
        # in the next graph
        return total_loss, (hidden_state.detach(), cell_state.detach())

class LMRNN(nn.Module):
    """
    Defines the core neural network for doing sentiment classification
    over a single datapoint at a time.

    It consists of an embedding average processing layer
    and multiple hidden layers.

    The forward() function does the important computation.
    The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, embedding_size, hidden_size, vocab_size, out_size):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param embedding_size: size of embedding (integer)
        :param hidden_size: size of hidden layer (integer)
        :param vocab_size: size of vocab (integer)
        :param out_size: size of output (integer), which should be the number of classes - 2 for vowels, consonants
        """
        super(LMRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

        self.hidden_to_out = nn.Linear(hidden_size, out_size)
        self.output_size = hidden_size

        # Initialize weights according to a formula due to Xavier Glorot.
        self.init_weights()

    """
    Initialize the LSTM weights
    Inspired from lstm_lecture.py, using Glorot
    """
    def init_weights(self):
        # This is a randomly initialized RNN.
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)

    def forward(self, x: torch.Tensor, prev_hidden_state=None, prev_cell_state=None):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: an array of character indices
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        if prev_hidden_state == None:
            # first arg is RNN num_layers*directions
            prev_hidden_state = torch.zeros(1, x.size()[0], LM_HIDDEN_SIZE)

        if prev_cell_state == None:
            # first arg is RNN num_layers*directions
            prev_cell_state = torch.zeros(1, x.size()[0], LM_HIDDEN_SIZE)

        # print(x.size())
        embedded = self.embedding(x)
        # print(embedded.size())
        sequence_hidden_state, (hidden_state, cell_state) = self.lstm(embedded, (prev_hidden_state, prev_cell_state))
        # print(hidden_state.size())

        return sequence_hidden_state, hidden_state, cell_state

LM_NUM_EPOCHS=10
LM_CHUNK_SIZE=20
# how much overlap we leave between chunks in a batch
LM_SKIP_SIZE=4
LM_BATCH_SIZE=LM_CHUNK_SIZE // LM_SKIP_SIZE

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    lm = RNNLanguageModel(vocab_index)

    # train_text = train_text[0:100]
    # dev_text = train_text[0:100]

    for epoch in range(LM_NUM_EPOCHS):
        lm.rnn.train()

        total_loss = 0.0
        # (num_layers * num_directions, batch, hidden_size)
        prev_hidden_state = None
        prev_cell_state = None

        for range_start in np.arange(0, len(train_text), LM_CHUNK_SIZE):

            contexts = []
            labels = []

            for i in np.arange(range_start, range_start + LM_CHUNK_SIZE, LM_SKIP_SIZE):
                range_end = i + LM_CHUNK_SIZE
                if range_end >= len(train_text):
                    break

                contexts.append(train_text[i:range_end])
                context_labels = [vocab_index.index_of(train_text[i+1]) for i in range(i, range_end)]
                labels.append(context_labels)

            if len(contexts) == 0:
                break

            index = (i % LM_CHUNK_SIZE)/LM_SKIP_SIZE
            loss, (prev_hidden_state, prev_cell_state) = lm.train(contexts, labels, prev_hidden_state, prev_cell_state)
            # TODO: should or should not reset state
            prev_hidden_state = None
            prev_cell_state = None
            total_loss += loss

        print("Total loss on epoch %i: %f" % (epoch, total_loss))

        lm.rnn.eval()
        evaluate = True

        # Evaluate on the dev set
        if evaluate:
            log_prob = lm.get_log_prob_sequence(dev_text, " ")
            avg_log_prob = log_prob/len(dev_text)
            perplexity = np.exp(-log_prob / len(dev_text))

            # import ipdb; ipdb.set_trace()

            print("Epoch %i dev - Log Prob %f. Perplexity: %f" % (epoch, log_prob, perplexity))
            pickle.dump(lm, open("lm.pickle", "wb"))

    return lm
