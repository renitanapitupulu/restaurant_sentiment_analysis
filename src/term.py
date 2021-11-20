from nltk import pos_tag
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import accuracy_score
from nltk import download


def encode_terms(tokens, terms):
    # Encode tokens to POS tag and BIO label
    encoded = []
    pos = pos_tag(tokens)
    for i in range(len(tokens)):
        label = 'O'
        for term in terms:
            tokenized_term = word_tokenize(term)
            for j in range(len(tokenized_term)):
                if tokens[i] == tokenized_term[j]:
                    if (j == 0):
                        label = 'B'
                    else:
                        label = 'I'
        encoded.append((tokens[i], pos[i][1], label))
    return encoded


def word2features(sent, i):
    # Convert each words in sent to features
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    # Convert words in sent to features

    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    # Get label from words in sent

    return [label for token, postag, label in sent]


def sent2tokens(sent):
    # Get token from words in sent

    return [token for token, postag, label in sent]


def aspect_term_extraction_model(tokens_list, terms_list):
    # Train model to extract aspect terms from tokens

    # Encode input
    encoded_terms_list = []
    for i in range(len(tokens_list)):
        encoded_terms = encode_terms(tokens_list[i], terms_list[i])
        encoded_terms_list.append(encoded_terms)

    data = [sent2features(s) for s in encoded_terms_list]
    target = [sent2labels(s) for s in encoded_terms_list]

    # Sequential labeling model
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               c1=0.1,
                               c2=0.1,
                               max_iterations=1000,
                               all_possible_transitions=True)
    crf.fit(data, target)
    return crf


def aspect_term_extraction(model, tokens_list, terms_list):
    # Predict aspect from tokens

    # Encode input
    encoded_terms_list = []
    for i in range(len(tokens_list)):
        encoded_terms = encode_terms(tokens_list[i], terms_list[i])
        encoded_terms_list.append(encoded_terms)

    data = [sent2features(s) for s in encoded_terms_list]
    target = [sent2labels(s) for s in encoded_terms_list]

    # Predict data
    pred = model.predict(data)
    # labels = list(model.classes_)
    # labels.remove('O')
    # print(metrics.flat_classification_report(
    #     target, pred, labels=labels, digits=3
    # ))

    # Decode label to aspect
    aspects_list = []
    for i in range(len(pred)):
        aspects = []
        c = 0
        for j in range(len(pred[i])):
            if pred[i][j] == 'B':
                aspects.append(tokens_list[i][j])
                c += 1
            elif pred[i][j] == 'I':
                if (c > 0):
                    aspects[c - 1] += ' ' + tokens_list[i][j]
                else:
                    aspects.append(tokens_list[i][j])
                    c += 1
        # print(aspects)
        aspects_list.append(aspects)
    return aspects_list