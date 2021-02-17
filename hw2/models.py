# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *

import pickle
import sys

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


OUTPUT_SIZE = 2

class NeuralSentimentClassifier(SentimentClassifier):

    def __init__(self, embeddings, lr):
        self.embeddings = embeddings
        input_size = embeddings.get_embedding_length()
        self.ffnn = SentimentNN(input_size, input_size, OUTPUT_SIZE, embeddings)
        self.optimizer = optim.Adam(self.ffnn.parameters(), lr=lr)

    def preprocess_input(self, sentences: List[List[str]]) -> torch.Tensor:
        """
        get embedding indexes for each word
        """
        length = max([len(sentence) for sentence in sentences])
        idx = lambda word: self.embeddings.word_indexer.index_of(word)
        pad_idx = idx("PAD")

        indexed_sentences = []
        for sentence in sentences:
            idxs = [idx(word) for word in sentence]
            idxs = [i if i != -1 else idx("UNK") for i in idxs]
            idxs += [pad_idx] * (length - len(sentence))
            indexed_sentences.append(torch.tensor(idxs))

        return torch.stack(indexed_sentences)

    def predict(self, ex_words: List[str]) -> int:
        """
        @see SentimentClassifier#predict
        """
        log_probs = self.ffnn.forward(self.preprocess_input([ex_words]))
        # from IPython import embed; embed()
        return torch.argmax(log_probs)

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        @see SentimentClassifier#predict_all
        """
        log_probs = self.ffnn.forward(self.preprocess_input(all_ex_words))
        return torch.argmax(log_probs, dim=1)

    def train(self, exs_words: List[List[str]], labels: List[int]) -> float:
        """
        NN train step
        NOTE: inspired from FFNN example

        returns training loss
        """
        x = self.preprocess_input(exs_words)
        # Build one-hot representation of y. Instead of the label 0 or 1, label_onehot is either [0, 1] or [1, 0]. This
        # way we can take the dot product directly with a probability vector to get class probabilities.
        label_onehot = torch.zeros(len(labels), OUTPUT_SIZE)
        indexes = torch.from_numpy(np.array(labels, dtype=np.int64).reshape(len(labels), 1))
        # scatter will write the value of 1 into the position of label_onehot given by y
        label_onehot.scatter_(1, indexes, 1)
        # Zero out the gradients from the torch.from_numpy(np.array(labels, dtype=np.int64))FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
        self.ffnn.zero_grad()
        log_probs = self.ffnn.forward(x)
        # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
        loss = torch.tensordot(torch.neg(log_probs), label_onehot)
        # Computes the gradient and takes the optimizer step
        loss.backward()
        self.optimizer.step()

        return loss

class SentimentNN(nn.Module):
    """
    Defines the core neural network for doing sentiment classification
    over a single datapoint at a time.

    It consists of an embedding average processing layer
    and multiple hidden layers.

    The forward() function does the important computation.
    The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid, out, embeddings):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(SentimentNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(embeddings.vectors).float(), freeze=True, padding_idx=0)
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        # TODO: multiple?
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.Tensor):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param sentence: an array of words
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        embedded = torch.mean(self.embedding(x), 1)
        # from IPython import embed; embed()
        return self.log_softmax(self.W(self.g(self.V(embedded))))

DROPOUT_RATE = 0.0

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    classifier = NeuralSentimentClassifier(word_embeddings, lr=args.lr)

    for epoch in range(0, args.num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for range_start in np.arange(0, len(ex_indices), args.batch_size):
            range_end = min(range_start + args.batch_size, len(ex_indices) - 1)
            if range_end == range_start:
                continue

            sliced_train_exs = train_exs[range_start:range_end]
            words = [ex.words for ex in sliced_train_exs]
            labels = [ex.label for ex in sliced_train_exs]
            loss = classifier.train(words, labels)
            total_loss += loss
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

        evaluate = True

        # Evaluate on the dev set
        if evaluate and epoch % 5 == 0:
            num_correct = 0
            num_total = 0
            for idx in range(0, len(dev_exs)):
                prediction = classifier.predict(dev_exs[idx].words)
                if dev_exs[idx].label == prediction:
                    num_correct += 1
                num_total += 1
            from IPython import embed; embed()
            acc = float(num_correct) / num_total
            print("Epoch %i dev - Accuracy: %i / %i = %f" % (epoch, num_correct, num_total, acc))
            pickle.dump(classifier, open("classifier.pickle", "wb"))

    return classifier


# Example of training a feedforward network with one hidden layer to solve XOR.
if __name__=="__main__":
    classifier = pickle.load(open("classifier.pickle", "rb"))
    sentence = sys.argv[1:]
    prediction = classifier.predict(sentence)
    print(f"Prediction is: {prediction}")

    log_probs = classifier.ffnn.forward(classifier.preprocess_input([sentence]))
    from IPython import embed; embed()
    assert(torch.exp(log_probs).sum() == 1.)
    print(f"Probs are: {torch.exp(log_probs)}")
