
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
download('punkt')
download('wordnet')
download('averaged_perceptron_tagger')


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


def split_df(original_df, category):
    entries = []

    for idx, data in original_df.iterrows():
        for aspect_idx, aspect in enumerate(data['aspects']):
            if (aspect == category):
                entry = {}
                numerical_class = 0
                if data['class'][aspect_idx] == 'positive':
                    numerical_class = 1
                elif data['class'][aspect_idx] == 'negative':
                    numerical_class = -1

                entry["text"], entry["aspects"], entry['class'] = data['text'], aspect, numerical_class
                entries.append(entry)

    return pd.DataFrame(entries)


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


def preprocess(df):
    # Preprocess data

    # Remove missing values
    df.dropna(inplace=True)

    # Tokenize
    df['tokenized'] = df['review'].apply(word_tokenize)
    df['preprocessed'] = df['tokenized'].apply(clean)
    df.drop(columns=['tokenized'], inplace=True)