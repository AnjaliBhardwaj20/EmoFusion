# -*- coding: utf-8 -*-

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from config import Config
import numpy as np
import logging

logger = logging.getLogger(__name__)


def run_classical_models(x_train, y_train, x_test, y_test, model='svm'):
    if model == 'svm':
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
    elif model == 'nb':
        clf = GaussianNB()
    elif model == 'gb':
        clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000)
    else:
        raise ValueError(f"Unsupported model type: {model}")

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    logger.info(f"\n{model.upper()} Classification Report:\n{classification_report(y_test, pred, digits=4)}")


def build_bilstm_model(input_size1, input_size2, output_size):
    model = Sequential([
        Bidirectional(LSTM(Config.BILSTM_UNITS_1, return_sequences=True, input_shape=(input_size1, input_size2))),
        Dropout(Config.DROPOUT_RATE),
        Bidirectional(LSTM(Config.BILSTM_UNITS_2)),
        Dropout(Config.DROPOUT_RATE),
        Dense(Config.DENSE_UNITS, activation='relu'),
        Dropout(Config.DROPOUT_RATE),
        Dense(output_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_bilstm_model(x_train, y_train, x_test, y_test, config):
    model = build_bilstm_model(config.MAX_LEN, config.EMBEDDING_SIZE, y_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=config.PATIENCE)

    logger.info("Training BiLSTM model...")
    history = model.fit(x_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                        validation_split=0.1, verbose=2, callbacks=[early_stop])

    logger.info("Evaluating BiLSTM model...")
    result = model.evaluate(x_test, y_test)
    pred_test = np.argmax(model.predict(x_test), axis=-1)
    true_labels = np.argmax(y_test, axis=1)

    logger.info("\nBiLSTM Classification Report:\n%s", classification_report(true_labels, pred_test, digits=4))