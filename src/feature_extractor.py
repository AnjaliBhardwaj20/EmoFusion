# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from afinn import Afinn
from config import Config
from preprocess import Preprocessor
import logging

logger = logging.getLogger(__name__)


def convert_glove(glove_input_file, word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)


def load_word_vectors(path):
    return KeyedVectors.load_word2vec_format(path, binary=False)


def load_all_lexicons():
    emolex = pd.read_csv(Config.EMOLEX, names=["word", "emotion", "association"], sep='\t', keep_default_na=False)
    emosensenet = pd.read_excel(Config.EMOSENSENET)
    nrc_emotion_intensity = pd.read_csv(Config.NRC_EIL, names=["word", "emotion", "score"], sep='\t', keep_default_na=False)
    nrc_vad = pd.read_csv(Config.NRC_VAD, sep='\t', keep_default_na=False)

    combined = pd.concat([emolex, emosensenet, nrc_vad, nrc_emotion_intensity], axis=0)

    return (emolex, emosensenet, nrc_emotion_intensity, nrc_vad), EMOTION_LABELS, combined


def analyze_similar_words(word, model):
    afinn = Afinn()
    try:
        word_list = [wrd for wrd, _ in model.most_similar(word, topn=25)]
    except:
        return []

    word_sentiments = {wrd: afinn.score(wrd) for wrd in word_list}
    word_sentiment = afinn.score(word)

    if 0 < word_sentiment <= 5:
        sorted_words = sorted([w for w, s in word_sentiments.items() if 0 < s <= 5], key=lambda x: word_sentiments[x], reverse=True)[:5]
    elif -5 <= word_sentiment < 0:
        sorted_words = sorted([w for w, s in word_sentiments.items() if -5 <= s < 0], key=lambda x: word_sentiments[x])[:5]
    else:
        sorted_words = [w for w, s in word_sentiments.items() if s == 0][:5]

    return sorted_words


def get_safe_value(df, word, col):
    try:
        return float(df[df['word'] == word][col].iloc[0])
    except:
        return 0.0


def process_top_five_words(word, lexicons, emotion_list, model):
    emolex, emosensenet, nrc_emotion_intensity, nrc_vad = lexicons
    word_vector = []

    for col in ['valence', 'arousal', 'dominance']:
        word_vector.append(get_safe_value(nrc_vad, word, col))

    intensity = nrc_emotion_intensity[nrc_emotion_intensity.word == word]
    word_vector.append(float(intensity.score.iloc[0]) if not intensity.empty else 0.0)

    for emotion in emotion_list:
        word_vector.append(get_safe_value(emolex, word, emotion))

    if word in emosensenet.Concepts.values:
        row = emosensenet[emosensenet.Concepts == word].iloc[0]
        word_vector.extend([float(row[emo]) for emo in emotion_list])
    else:
        word_vector.extend([0.0] * 6)

    for w in analyze_similar_words(word, model):
        for col in ['valence', 'arousal', 'dominance']:
            word_vector.append(get_safe_value(nrc_vad, w, col))
        intensity = nrc_emotion_intensity[nrc_emotion_intensity.word == w]
        word_vector.append(float(intensity.score.iloc[0]) if not intensity.empty else 0.0)
        for emotion in emotion_list:
            word_vector.append(get_safe_value(emolex, w, emotion))
        if w in emosensenet.Concepts.values:
            row = emosensenet[emosensenet.Concepts == w].iloc[0]
            word_vector.extend([float(row[emo]) for emo in emotion_list])
        else:
            word_vector.extend([0.0] * 6)

    word_vector.extend([0.0] * ((5 - len(analyze_similar_words(word, model))) * 16))
    return word_vector


def prepare_features(texts, model, lexicons, emotion_list, combined):
    train_vocab = set(word for text in texts for word in str(text).split())
    glove_vocab = model.index_to_key

    embedded = []
    for sentence in texts:
        doc = []
        for word in str(sentence).split():
            if word in train_vocab and word in glove_vocab:
                word_vect = list(model[word])
                if word in combined.word.unique():
                    sim_words = analyze_similar_words(word, model)
                    sim_vecs = [list(model[sim]) for sim in sim_words if sim in glove_vocab]
                    if sim_vecs:
                        word_vect = list(np.add(word_vect, np.mean(sim_vecs, axis=0)))
                word_vect.extend(process_top_five_words(word, lexicons, emotion_list, model))
                doc.append(word_vect)
        if doc:
            doc_vec = np.mean(np.vstack(doc), axis=0)
        else:
            doc_vec = np.zeros(196)
        embedded.append(doc_vec)
    return np.nan_to_num(np.vstack(embedded))
