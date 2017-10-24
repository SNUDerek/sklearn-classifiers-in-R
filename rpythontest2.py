# test variable and test function:
test_string = "hello world"

import codecs, re
import numpy as np
# preprocessing
# from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
# classifiers
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# model pipeline stuff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from konlpy.tag import Mecab

def printthis():
    return ("hello world")


# mecab-tokenize a string
def mecab_tokenize(sentence):
    mecab = Mecab()
    return mecab.morphs(sentence)


# mecab-tokenize a list of strings
def mecab_tokenize_list(sents):
    mecab = Mecab()
    sent_toks = []
    # POS-tag and get lexical form from morphemes using KONLPY
    for sent in sents:
        sent_toks.append(mecab.morphs(sent))
    return sent_toks


# encode labels
def encodelabels(labels):
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return labels


# FIT a classifier on training data
# callable from R with reticulate
def r_classify(sents, labels):

    # listify to prevent any wonky reticulate JSON errors
    sents = list(sents)
    labels = list(labels)

    # encode labels
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    # define model and train
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=mecab_tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])

    pipeline.fit(sents, labels)

    # pack model and encoder into list
    return [pipeline, encoder]


# PREDICT using a classifier on larger corpus
# callable from R with reticulate
def r_predict(packedtuple, sents, labels):

    # unpack model and encoder
    pipeline = packedtuple[0]
    encoder = packedtuple[1]

    # listify to prevent any wonky reticulate reticulate JSON errors
    sents = list(sents)
    labels = list(labels)

    # encode labels
    labels = encoder.transform(labels)

    # predict on model
    preds = pipeline.predict(sents)
    preds = [encoder.inverse_transform(i) for i in preds]

    # return AS NUMPY ARRAY!!
    return np.asarray(preds)




