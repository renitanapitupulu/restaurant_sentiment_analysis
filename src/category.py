from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#!/usr/bin/python
# -*- coding: utf-8 -*-


def preprocess_term_train(train):
    aspect_term = []
    aspect_ctgr = []
    for i in range(len(train)):

        # aspect_term = []
        # aspect_ctgr = []

        if len(train['terms'][i]) > 0:
            term_len = len(train['terms'][i])

            # print(term_len)

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
                        aspect_ctgr.append(train['aspects'][i][(l - 1) // k])
        else:
            aspect_term.append('')
            aspect_ctgr.append('anecdotes/miscellaneous')
    return (aspect_term, aspect_ctgr)


def aspect_categorization_model(aspect_term, aspect_ctgr):
    def dummy(doc):
        return doc

    tfidf = TfidfVectorizer(
        analyzer='word',  # ''
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None)

    # transform term to numerical features using the trained model
    X_train = tfidf.fit_transform(aspect_term).toarray()
    # X_train = tfidf.fit_transform(train['terms']).toarray()

    # convert categorical aspect class to numerical
    # mlb = MultiLabelBinarizer()
    # y_train = mlb.fit_transform(train['aspects'])
    le = LabelEncoder()
    y_train = le.fit_transform(aspect_ctgr)

    # train the model clasification
    # clf = LabelPowerset(MultinomialNB(alpha=1e-1))
    clf = RandomForestClassifier(random_state=0, n_estimators=50)
    clf.fit(X_train, y_train)

    return clf, le, tfidf


def aspect_categorization(ctgr_model, le_model, tfidf_model, term_aspect):
    term_prep = []
    for i in range(len(term_aspect)):
        if len(term_aspect[i]) > 0:
            for j in range(len(term_aspect[i])):
                term_prep.append(term_aspect[i][j])
        else:
            term_prep.append('')

    X_test = tfidf_model.transform(term_prep).toarray()
    aspect_pred = ctgr_model.predict(X_test)
    y_pred = le_model.inverse_transform(aspect_pred)
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
    # pred_df = pd.DataFrame({'review': test['review'], 'aspects': categories})
    return categories


# term_data, ctgr_data = preprocess_term_train(train)
# aspect_ctgr_model, le_model, tfidf_model = aspect_categorization_model(
#     term_data, ctgr_data)
# aspect_pred_df = aspect_categorization(
#     aspect_ctgr_model, le_model, tfidf_model, term_aspect)
