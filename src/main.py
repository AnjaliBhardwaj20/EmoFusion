# -*- coding: utf-8 -*-

import argparse
import logging
from config import Config
from feature_extractor import convert_glove, load_word_vectors, load_all_lexicons, prepare_features
from models.classifiers import run_classical_models, run_bilstm_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Emotion Classification CLI Tool")
    parser.add_argument('--model', choices=['svm', 'nb', 'gb', 'bilstm'], required=True,
                        help="Choose which model to run: svm, nb, gb, bilstm")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")

    df = pd.read_csv(Config.DATA_PATH)
    Y = df["label"]    
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in splitter.split(df.preprocess_text, Y):
        X_train_texts, X_test_texts = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    logger.info("Preparing embeddings and lexicons...")
    convert_glove(Config.GLOVE_PATH, Config.GENSIM_GLOVE_PATH)
    word2vec_model = load_word_vectors(Config.GENSIM_GLOVE_PATH)
    lexicons, emotion_list, combined_df = load_all_lexicons()

    logger.info("Generating feature vectors...")
    X_train = prepare_features(X_train_texts, word2vec_model, lexicons, emotion_list, combined_df)
    X_test = prepare_features(X_test_texts, word2vec_model, lexicons, emotion_list, combined_df)

    if args.model == 'bilstm':
        from tensorflow.keras.utils import to_categorical
        Y_cat = to_categorical(df["label"])
        Y_train_cat = Y_cat[Y_train.index]
        Y_test_cat = Y_cat[Y_test.index]
        run_bilstm_model(X_train, Y_train_cat, X_test, Y_test_cat, Config)
    else:
        run_classical_models(X_train, Y_train, X_test, Y_test, model=args.model)


if __name__ == '__main__':
    main()