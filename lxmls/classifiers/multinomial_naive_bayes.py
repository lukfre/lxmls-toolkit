import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        print("classes", classes)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        print("n_classes", n_classes)

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # ----------
        # Solution to Exercise 1

        # axis 0: doc id
        # axis 1: bag-of-words -> value at index i is the occurrence of word i
        print("x", x.shape)
        print("y", y.shape)
        neg_docs_ids, _ = np.nonzero(y == 0)
        pos_docs_ids, _ = np.nonzero(y == 1)

        prior[0] = len(neg_docs_ids) / n_docs
        prior[1] = len(pos_docs_ids) / n_docs
        print("prior", prior)

        # compute the frequency of words across the documents (axis 0)
        neg_docs_words_frequency = x[neg_docs_ids, :].sum(0)
        pos_docs_words_frequency = x[pos_docs_ids, :].sum(0)

        # the likelihood is the relative frequence of the words in the docs belonging to a given class
        #   -> just divide by the total number of words in the documents in that class
        likelihood[:, 0] = (1 + neg_docs_words_frequency) / (
            x[neg_docs_ids, :].sum() + n_words
        )
        likelihood[:, 1] = (1 + pos_docs_words_frequency) / (
            x[pos_docs_ids, :].sum() + n_words
        )

        # raise NotImplementedError("Complete Exercise 1")

        # End solution to Exercise 1
        # ----------

        params = np.zeros((n_words + 1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
