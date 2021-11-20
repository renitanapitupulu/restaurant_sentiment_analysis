from category import *
from preprocess import *
from term import *
from sentiment import *
from sentence import Sentence
import pandas as pd

if __name__ == "__main__":
    train = read_dataset('./data/Restaurants_Train.xml')
    test = read_dataset('./data/restaurants-trial.xml')
    print(train)
    preprocess(train)
    preprocess(test)

    ATE_model = aspect_term_extraction_model(train['preprocessed'],
                                             train['terms'])
    term_aspect = aspect_term_extraction(ATE_model, test['preprocessed'],
                                         test['terms'])

    term_data, ctgr_data = preprocess_term_train(train)
    aspect_ctgr_model, le_model, tfidf_model = aspect_categorization_model(
        term_data, ctgr_data)
    aspect_pred_df = aspect_categorization(aspect_ctgr_model, le_model,
                                           tfidf_model, term_aspect)
    print(aspect_pred_df)
