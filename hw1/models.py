# models.py

from sentiment_data import *
from utils import *

import numpy as np

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        @see FeatureExtractor#extract_features

        TODO: do you want to throw out low-count words? Do you want to lowercase? Do you want to discard
        stopwords? Do you want the value in the feature vector to be 0/1 for absence or presence of a word, or
        reflect its count in the given sentence?
        """
        counter = Counter()
        for gram in sentence:
            counter[gram] += 1
            if add_to_indexer:
                self.indexer.add_and_get_index(gram)

        return counter

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        raise Exception("Must be implemented")

    def get_indexer(self):
        """
        @see FeatureExtractor#get_indexer
        """
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        @see FeatureExtractor#extract_features
        """
        return self.indexer

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, featurizer):
        self.featurizer = featurizer
        self.indexer = featurizer.get_indexer()
        self.w = np.array([])


    def train(self, sentence: List[str], label: int, alpha: float, add_new_features=False):
        sparse_features = self.featurizer.extract_features(sentence, add_to_indexer=add_new_features)

        if add_new_features:
            self.w.resize(len(self.indexer))

        predicted_label = self.predict_from_features(sparse_features)
        if label != predicted_label:
            for gram, value in sparse_features.items():
                self.w[self.indexer.index_of(gram)] += alpha * value * (label - predicted_label)


    def predict(self, sentence: List[str]) -> int:
        sparse_features = self.featurizer.extract_features(sentence)

        return self.predict_from_features(sparse_features)

    def predict_from_features(self, features: Counter) -> int:
        w_gram = lambda gram : self.w[self.indexer.index_of(gram)] if self.indexer.contains(gram) else 0
        features_times_w = sum([value*w_gram(gram) for gram, value in features.items()])

        if features_times_w > 0:
            return 1
        return 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    classifier = PerceptronClassifier(feat_extractor)

    # epochs?
    for t in range(75):
        random_is = np.arange(len(train_exs))
        np.random.shuffle(random_is)
        for i in random_is:
            example = train_exs[i]
            classifier.train(example.words, example.label, alpha=1.0/(t+1), add_new_features=(t==0))

    return classifier


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

def test_perceptron():
    feat_extractor = UnigramFeatureExtractor(Indexer())

    classifier = PerceptronClassifier(feat_extractor)

    assert(classifier.predict(["good"]) == 0)
    classifier.train(["good"], 1, alpha=1, add_new_features=True)
    assert(classifier.predict(["good"]) == 1)
    assert(classifier.predict(["not", "good"]) == 1)

    classifier.train(["bad", "bad", "not", "good"], 0, alpha=1, add_new_features=True)
    assert(classifier.predict(["not", "good"]) == 0)

def test_unigrams():
    feat_extractor = UnigramFeatureExtractor(Indexer())

    counts = feat_extractor.extract_features(["taco", "cat", "taco"])
    assert(counts["taco"] == 2)
    assert(counts["cat"] == 1)

    feat_extractor.extract_features(["I", "ai", "n't", "a", "cat", "."], add_to_indexer=True)
    feat_extractor.extract_features(["I", "do", "n't", "have", "face", "."], add_to_indexer=True)

    indexer = feat_extractor.get_indexer()
    assert(len(indexer) == 9)

if __name__ == '__main__':
    test_unigrams()
    test_perceptron()
