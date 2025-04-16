# -*- coding: utf-8 -*-

class Config:
    DATA_PATH = "data/dataset_name.csv"
    GLOVE_PATH = "embeddings/glove.6B.100d.txt"
    GENSIM_GLOVE_PATH = "embeddings/gensim_glove.6B.100d.txt"
    EMOLEX = "data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    EMOSENSENET = "data/emosenticnet.xls"
    NRC_EIL = "data/NRC-Emotion-Intensity-Lexicon-v1.txt"
    NRC_VAD = "data/NRC-VAD-Lexicon.txt"
    
    
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    MAX_LEN = 50 
    EMBEDDING_SIZE = 196
    BATCH_SIZE = 32
    EPOCHS = 15
    PATIENCE = 2

    # BiLSTM parameters
    BILSTM_UNITS_1 = 196
    BILSTM_UNITS_2 = 98
    DENSE_UNITS = 46
    DROPOUT_RATE = 0.2
