# models.py

from sentiment_data import *
from utils import *

from functools import reduce

import numpy as np


from nltk import skipgrams
from nltk import text
#  nltk.download('stopwords')
#  from nltk.corpus import stopwords
#  en_stopwords = set(stopwords.words('english'))
# NOTE: top stopwords to filter out; work better than the nltk stopwords (which include some negations seemingly)
TOP_STOPWORDS = [".", ",", "the", "and", "of", "a", "to", "is", "'s", "that",
        "in", "it", "The", "as", "an", "on", "by", "so"]
PUNCTUATION = [".", ",", ";", "?", "!", ":", "---", "..."]

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def preprocess_sentence(self, sentence):
        preprocessed_sentence = []

        for word in sentence:
            # lowercasing makes a slight difference
            word = word.lower()
            if word in TOP_STOPWORDS or word in PUNCTUATION:
                continue
            preprocessed_sentence.append(word)

        return preprocessed_sentence

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
        """
        counter = Counter()
        for gram in self.preprocess_sentence(sentence):
            # I also tried using 1/0 vs counts, similar result
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

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        @see FeatureExtractor#extract_features
        """
        counter = Counter()
        preprocessed_sentence = self.preprocess_sentence(sentence)

        for i1, i2 in zip(range(0, len(preprocessed_sentence)-1), range(1, len(preprocessed_sentence))):
            gram = f"{preprocessed_sentence[i1]}|{preprocessed_sentence[i2]}"
            # I also tried using 1/0 vs counts, similar result
            counter[gram] += 1
            if add_to_indexer:
                self.indexer.add_and_get_index(gram)

        return counter

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, text_collection):
        self.indexer = indexer
        self.text_collection = text_collection

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        @see FeatureExtractor#extract_features
        """
        counter = Counter()
        negated_sentence = []

        negating = False

        # negative tags - marginally helpful?
        for gram in sentence:
            if gram in PUNCTUATION:
                negating = False

            # add negative tags for the rest of the sentence
            if negating:
                negated_sentence.append("NOT_" + gram)
            else:
                negated_sentence.append(gram)

            if gram in ["n't", "not"]:
                negating = True

        #  preprocessed_sentence = self.preprocess_sentence(sentence)
        preprocessed_sentence = self.preprocess_sentence(negated_sentence)

        # + TRIGRAMS
        for num_grams in range(1, 3):
            for i in range(len(preprocessed_sentence)-num_grams):
                words = preprocessed_sentence[slice(i, i+num_grams)]
                gram = "|".join(words)

                weights = [self.text_collection.tf_idf(word, preprocessed_sentence) for word in words]
                score = reduce(lambda x, y: x + y, weights)
                # clip low score - common words
                score = 0.0 if score < 0.2 else score
                counter[gram] = score

                if add_to_indexer:
                    self.indexer.add_and_get_index(gram)

        # NOTE: skipgrams did not help...
        #  for grams in skipgrams(preprocessed_sentence, 3, 2):
        #      gram = "|".join(grams)
        #      # I also tried using 1/0 vs counts, similar result
        #      counter[gram] += 1
        #      if add_to_indexer:
        #          self.indexer.add_and_get_index(gram)

        return counter


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
        features = self.featurizer.extract_features(sentence, add_to_indexer=add_new_features)

        if add_new_features:
            self.w.resize(len(self.indexer))

        predicted_label = self.predict_from_features(features)
        if label != predicted_label:
            for gram, value in features.items():
                self.w[self.indexer.index_of(gram)] += alpha * value * (label - predicted_label)


    def predict(self, sentence: List[str]) -> int:
        features = self.featurizer.extract_features(sentence)

        return self.predict_from_features(features)

    def predict_from_features(self, features: Counter) -> int:
        w_gram = lambda gram : self.w[self.indexer.index_of(gram)] if self.indexer.contains(gram) else 0
        w_dot_fx = sum([value*w_gram(gram) for gram, value in features.items()])

        if w_dot_fx > 0:
            return 1
        return 0

class LogisticRegressionClassifier(SentimentClassifier):
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
        features = self.featurizer.extract_features(sentence, add_to_indexer=add_new_features)

        if add_new_features:
            self.w.resize(len(self.indexer))

        p_1 = self.positive_probability_of_features(features)

        for gram, fx in features.items():
            # (stochastic) gradient update
            if label == 1:
                self.w[self.indexer.index_of(gram)] += alpha * fx * (1 - p_1)
            else:
                self.w[self.indexer.index_of(gram)] -= alpha * fx * (p_1)


    def predict(self, sentence: List[str]) -> int:
        features = self.featurizer.extract_features(sentence)

        if self.positive_probability_of_features(features) > 0.5:
            return 1
        return 0

    def positive_probability_of_features(self, features: Counter):
        w_gram = lambda gram : self.w[self.indexer.index_of(gram)] if self.indexer.contains(gram) else 0
        w_dot_fx = sum([fx*w_gram(gram) for gram, fx in features.items()])
        exp_w_dot_fx = np.exp(w_dot_fx)
        return exp_w_dot_fx / (1 + exp_w_dot_fx)

    def log_likelihood(self, exs: List[SentimentExample]):
        sum = 0

        for ex in exs:
            features = self.featurizer.extract_features(ex.words)
            sum += self.positive_probability_of_features(features)

        return sum

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    classifier = PerceptronClassifier(feat_extractor)

    # TODO: early stopping?
    for t in range(25):
        random_is = np.arange(len(train_exs))
        np.random.shuffle(random_is)
        for i in random_is:
            example = train_exs[i]
            classifier.train(example.words, example.label, alpha=1.0/(t+1), add_new_features=(t==0))

    return classifier


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs=30, alpha=.5) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    classifier = LogisticRegressionClassifier(feat_extractor)

    # TODO: early stopping?
    for t in range(num_epochs):
        random_is = np.arange(len(train_exs))
        np.random.shuffle(random_is)
        for i in random_is:
            example = train_exs[i]
            train_alpha = 1.0/(t+1) if alpha == 'inverse' else alpha
            classifier.train(example.words, example.label, alpha=train_alpha, add_new_features=(t==0))

    return classifier


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
        tc = text.TextCollection([ex.words for ex in train_exs])
        feat_extractor = BetterFeatureExtractor(Indexer(), tc)
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

def test_logistic_regression():
    feat_extractor = UnigramFeatureExtractor(Indexer())

    classifier = LogisticRegressionClassifier(feat_extractor)

    assert(classifier.predict(["good"]) == 0)
    classifier.train(["good"], 1, alpha=1, add_new_features=True)
    assert(classifier.predict(["good"]) == 1)
    assert(classifier.predict(["not", "good"]) == 1)

    classifier.train(["bad", "bad", "not", "good"], 0, alpha=1, add_new_features=True)
    classifier.train(["bad", "bad", "not", "good"], 0, alpha=1, add_new_features=True)
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
    assert(len(indexer) == 7)

def test_bigrams():
    feat_extractor = BigramFeatureExtractor(Indexer())

    counts = feat_extractor.extract_features(["taco", "cat", "taco"])
    #  assert(counts["taco"] == 2)
    assert(counts["taco|cat"] == 1)
    #  assert(counts["cat"] == 1)
    assert(counts["cat|taco"] == 1)

    feat_extractor.extract_features(["I", "ai", "n't", "a", "cat", "."], add_to_indexer=True)
    feat_extractor.extract_features(["I", "do", "n't", "have", "face", "."], add_to_indexer=True)

    indexer = feat_extractor.get_indexer()
    # BIGRAM + NOT tags
    #  assert(len(indexer) == 10)
    # regular BIGRAM
    assert(len(indexer) == 7)
    # UNIGRAM + BIGRAM
    #  assert(len(indexer) == 17)

if __name__ == '__main__':
    test_unigrams()
    test_bigrams()
    test_perceptron()
    test_logistic_regression()
