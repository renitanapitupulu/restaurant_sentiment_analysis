

import pickle
import re
from sklearn.linear_model import RidgeClassifier
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk import download
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import accuracy_score

# Read dataset


def read_dataset(file):
    tree = ET.parse(file)
    root = tree.getroot()

    dataset = []
    for sentence in root.findall("sentence"):
        data = {}
        terms = []
        aspects = []
        polarity = []
        if sentence.find("aspectTerms"):
            for term in sentence.find("aspectTerms").findall("aspectTerm"):
                terms.append(term.get("term"))
        if sentence.find("aspectCategories"):
            for aspect in sentence.find("aspectCategories").findall("aspectCategory"):
                aspects.append(aspect.get("category"))
                polarity.append(aspect.get("polarity"))
        data["review"] = sentence[0].text
        data["terms"] = terms
        data["aspects"] = aspects
        data["polarity"] = polarity
        dataset.append(data)
    return pd.DataFrame(dataset)


train = read_dataset('data/Restaurants_Train.xml')
test = read_dataset('data/restaurants-trial.xml')
train.head()

# # Run once when starting runtime
download('punkt')
download('wordnet')
download('averaged_perceptron_tagger')


def map_pos_tag(word):
    # Map POS tag to lemmatize pos parameter

    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean(tokens):
    # Clean list of tokens

    # Convert to lower case
    cleaned = [w.lower() for w in tokens]

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    cleaned = [w.translate(table) for w in cleaned]

    # Remove nonalphabetic
    cleaned = [w for w in cleaned if w.isalpha()]

    # Lemmatize
    # lemmatizer = WordNetLemmatizer()
    # lemmatized = [lemmatizer.lemmatize(w, map_pos_tag(w)) for w in cleaned]

    return cleaned


def handle_negation(review):
    negations_ = {"isn't": "is not", "can't": "can not", "couldn't": "could not", "hasn't": "has not",
                  "hadn't": "had not", "won't": "will not",
                  "wouldn't": "would not", "aren't": "are not",
                  "haven't": "have not", "doesn't": "does not", "didn't": "did not",
                  "don't": "do not", "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
                  "mightn't": "might not",
                  "mustn't": "must not"}
    negation_pattern = re.compile(
        r'\b(' + '|'.join(negations_.keys()) + r')\b')
    neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], review)
    return neg_handled


def preprocess(df):
    # Preprocess data

    # Remove missing values
    df.dropna(inplace=True)

    # Tokenize
    # handled_negation = df['review'].apply(handle_negation)
    df['tokenized'] = df['review'].apply(word_tokenize)
    df['preprocessed'] = df['tokenized'].apply(clean)
    df.drop(columns=['tokenized'], inplace=True)


preprocess(train)
preprocess(test)
train.head()


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
# CRF Features


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
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
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
    crf = sklearn_crfsuite.CRF(
        algorithm='ap',
        max_iterations=1000,
        all_possible_transitions=True
    )
    crf.fit(data, target)
    return crf


ATE_model = aspect_term_extraction_model(train['preprocessed'], train['terms'])
# pick = open("ATE_model.pickle", "wb")
# pickle.dump(ATE_model, pick)
# pick.close()


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
    labels = list(model.classes_)
    labels.remove('O')
    print(metrics.flat_classification_report(
        target, pred, labels=labels, digits=3
    ))

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


term_aspect = aspect_term_extraction(
    ATE_model, test['preprocessed'], test['terms'])
term_aspect

# preprocess term & aspect data to build the model


def preprocess_term_train(train):
    aspect_term = []
    aspect_ctgr = []
    for i in range(len(train)):
        if len(train['terms'][i]) > 0:
            term_len = len(train['terms'][i])
            aspect_len = len(train['aspects'][i])
            if term_len <= aspect_len:
                for j in range(term_len):
                    aspect_term.append(train['terms'][i][j])
                    aspect_ctgr.append(train['aspects'][i][j])
            else:
                k = term_len // aspect_len
                aspect_term.append(train['terms'][i][0])
                aspect_ctgr.append(train['aspects'][i][0])
                if term_len % aspect_len == 0:
                    for j in range(1, term_len):
                        aspect_term.append(train['terms'][i][j])
                        aspect_ctgr.append(train['aspects'][i][j // k])
                else:
                    for j in range(1, term_len):
                        l = j
                        aspect_term.append(train['terms'][i][j])
                        aspect_ctgr.append(train['aspects'][i][(l - 1)
                                                               // k])
        else:
            aspect_term.append('')
            aspect_ctgr.append('anecdotes/miscellaneous')
    return aspect_term, aspect_ctgr


term_data, ctgr_data = preprocess_term_train(train)
aspect_term = term_data
aspect_ctgr = ctgr_data


def dummy(doc):
    return doc


tfidf = TfidfVectorizer(
    analyzer='word',  # ''
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None)
# tfidf_m = open("tfidf_model.pickle", "wb")
# pickle.dump(tfidf, tfidf_m)
# tfidf_m.close()

# transform term to numerical features using the trained model
X_train = tfidf.fit_transform(aspect_term).toarray()

# convert categorical aspect class to numerical
le = LabelEncoder()
le.fit(aspect_ctgr)

aspect_ctgr1 = [[aspect_ctgr[i]] for i in range(len(aspect_ctgr))]
enc = OneHotEncoder(handle_unknown='ignore')
# enc_m = open("enc_model.pickle", "wb")
# pickle.dump(enc, enc_m)
# enc_m.close()
y_train = enc.fit_transform(aspect_ctgr1).toarray()

# build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        1024, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# train the model
history = model.fit(
    X_train, y_train, verbose=2, epochs=40,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='auto',
            verbose=1,
            baseline=None,
            restore_best_weights=True
        )
    ]
)


# def aspect_categorization_model(aspect_term, aspect_ctgr):


#     return model, le, tfidf, enc

aspect_ctgr_model = model
le_model = le
tfidf_model = tfidf
enc_model = enc
# aspect_ctgr_model, le_model, tfidf_model, enc_model = aspect_categorization_model(
#     term_data, ctgr_data)

# aspect_ctgr_model.save("aspect_ctgr_model.pickle")

# le = open("le_model.pickle", "wb")
# pickle.dump(le_model, le)
# le.close()


# predict the aspect category clasification for testing data


def aspect_categorization(ctgr_model, le_model, tfidf_model, enc_model, term_aspect, data_test):
    term_prep = []
    for i in range(len(term_aspect)):
        if len(term_aspect[i]) > 0:
            for j in range(len(term_aspect[i])):
                term_prep.append(term_aspect[i][j])
        else:
            term_prep.append('')

    X_test = tfidf_model.transform(term_prep).toarray()
    aspect_pred = ctgr_model.predict(X_test)

    # transform class to category
    y_pred = enc_model.inverse_transform(aspect_pred)
    categories = []
    k = 0
    for i in range(len(term_aspect)):
        category = []
        if len(term_aspect[i]) > 0:
            for j in range(len(term_aspect[i])):
                if y_pred[k] not in category:
                    category.append(y_pred[k])
                k += 1
        else:
            category.append(y_pred[k])
            k += 1
        categories.append(category)

    final_categories = []
    for i in range(len(categories)):
        cat = []
        for j in range(len(categories[i])):
            cat.append(categories[i][j][0])
        final_categories.append(cat)

    pred_df = pd.DataFrame({'review': data_test, 'aspects': final_categories})
    return pred_df


aspect_pred_df = aspect_categorization(aspect_ctgr_model, le_model,
                                       tfidf_model, enc_model, term_aspect, test['review'])

aspect_pred_df['term_aspect'] = term_aspect
aspect_pred_df['preprocessed'] = test['preprocessed']

train_df = train
test_df = aspect_pred_df
test_df['polarity'] = test['polarity']
aspects = [aspect[0] for aspect in train_df['aspects']]
train_df['out_aspect'] = aspects

unique_aspects = train_df['out_aspect'].unique()
test_df


def split_df(original_df, category, predict=True):
    entries = []

    for idx, data in original_df.iterrows():
        for aspect_idx, aspect in enumerate(data['aspects']):
            if (aspect == category):
                entry = {}
                numerical_class = 0
                if (predict):
                    if aspect_idx >= 0 and aspect_idx < len(data['polarity']):
                        if data['polarity'][aspect_idx] == 'positive':
                            numerical_class = 1
                        elif data['polarity'][aspect_idx] == 'negative':
                            numerical_class = -1

                    entry["text"], entry["aspects"], entry['polarity'], entry[
                        'preprocessed'] = data['review'], aspect, numerical_class, data['preprocessed']
                    entries.append(entry)
                else:
                    entry["text"], entry["aspects"], entry['preprocessed'] = data['review'], aspect, data['preprocessed']
                    entries.append(entry)
    return pd.DataFrame(entries)


aspect_dfs = {}
for aspect in unique_aspects:
    aspect_dfs[aspect] = split_df(train_df, aspect)

test_aspect_dfs = {}
for aspect in unique_aspects:
    test_aspect_dfs[aspect] = split_df(test_df, aspect)


def transform_sentence(df_train, df_test, predict=True):
    def dummy(doc):
        return doc

    # memanggil tfidfVectorizer dengan tokenizer dummy dan preprocessor dummy
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None)

    # transform setiap dataset dengan tfidf
    X_train = tfidf.fit_transform(df_train['preprocessed'])
    X_test = tfidf.transform(df_test['preprocessed'])
    y_train = df_train['polarity']
    y_test = df_test['polarity']
    return X_train, y_train, X_test, y_test, tfidf


vec_train = {}
targets = {}
vec_test = {}
target_test = {}
sa_tfidf = {}

for key in unique_aspects:
    vec_train[key], targets[key], vec_test[key], target_test[key], sa_tfidf[key] = transform_sentence(
        aspect_dfs[key], test_aspect_dfs[key])


def build_model(kernel, gamma, X_train, y_train, bool_svc=True):
    if (bool_svc):
        my_model = SVC(kernel='poly', gamma=1)
        my_model.fit(X_train, y_train)
    else:
        my_model = RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                                   max_iter=None, normalize=False, random_state=None, solver='auto',
                                   tol=0.001)
        my_model.fit(X_train, y_train)

    return my_model


def analyze_model(X_train, y_train, X_test, y_test):
    kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']
    gammas = [0.1, 1, 10, 100]

    # Iterate through kernel, iterate through gammas, find model with best accuracy score
    for kernel in kernel_type:
        for gamma in gammas:
            svc = SVC(kernel=kernel, gamma=gamma)
            svc.fit(X_train, y_train)
            train_pred = svc.predict(X_test)
            print(
                f'Using kernel = {kernel}, gamma = {gamma}, Accuracy: {accuracy_score(y_test, train_pred)}')


def build_ann(data_len):
    model = Sequential()
    model.add(Embedding(data_len, 300, input_length=data_len))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.005), metrics=['accuracy'])
    model.summary()
    return model


model = {}
gamma = {
    "service": 1,
    "price": 1,
    "ambience": 1,
    "anecdotes/miscellaneous": 10,
    "food": 10,
}
for key in unique_aspects:
    model[key] = build_model(
        'poly', gamma[key], vec_train[key], targets[key], False)

pred = {}
for key in unique_aspects:
    pred[key] = model[key].predict(vec_test[key])
    print("Accuracy: ", key)
    print(accuracy_score(target_test[key], pred[key]))


def predict_aspect_term(model, review):
    negations_ = {"isn't": "is not", "can't": "can not", "couldn't": "could not", "hasn't": "has not",
                  "hadn't": "had not", "won't": "will not",
                  "wouldn't": "would not", "aren't": "are not",
                  "haven't": "have not", "doesn't": "does not", "didn't": "did not",
                  "don't": "do not", "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
                  "mightn't": "might not",
                  "mustn't": "must not"}
    negation_pattern = re.compile(
        r'\b(' + '|'.join(negations_.keys()) + r')\b')
    neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], review)
    tokens = clean(word_tokenize(neg_handled))
    tokens_list = [tokens]
    encoded = []
    pos = pos_tag(tokens)
    data = [sent2features(s) for s in [pos]]
    pred = model.predict(data)
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
    return aspects_list, tokens_list


# All Function for predict


def predict_review(review):
    term_aspect, preprocessed = predict_aspect_term(ATE_model, review)
    #   term_aspect = aspect_term_extraction(ATE_model, testing_data['preprocessed'], testing_data['preprocessed'])
    ctgry = aspect_categorization(aspect_ctgr_model, le_model,
                                  tfidf_model, enc_model, term_aspect, [review])
    ctgry['preprocessed'] = preprocessed

    test_aspect_dfs = {}
    for asp in ctgry['aspects'][0]:
        test_aspect_dfs[asp] = split_df(ctgry, asp, False)
    # print(ctgry['preprocessed'])
    vect_predict = {}
    for asp in ctgry['aspects'][0]:
        vect_predict[asp] = sa_tfidf[asp].transform(ctgry['preprocessed'])
    predict = {}

    for key in ctgry['aspects'][0]:
        predict[key] = model[key].predict(vect_predict[key])

    return term_aspect[0], ctgry['aspects'][0], predict


# while(True):
#     review = input("Text: ")
#     asp, ctgr, tss = predict_review(review)
#     print("ASPECT:", asp)
#     print("CATEGORY:", ctgr)
#     print("SENTIMENT:", tss)
