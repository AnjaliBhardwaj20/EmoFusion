# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
import re
from pycontractions import Contractions
from nltk import download
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required resources
download('stopwords')
download('wordnet')


class Preprocessor:
    def __init__(self, dataframe: pd.DataFrame, text_column: str):
        self.df = dataframe.copy()
        self.text_column = text_column
        self.tokenizer = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.contraction_expander = pycontractions.Contractions(api_key="glove-twitter-100")

    def extract_hashtags(self):
        logging.info("Extracting hashtags...")
        self.df['hashtags'] = self.df[self.text_column].apply(lambda x: re.findall(r"#(\\w+)", str(x)))

    def remove_usernames(self):
        logging.info("Replacing Twitter handles with placeholder...")
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: re.sub(r'@[\S]+', '[NAME]', str(x)))

    def lowercase_text(self):
        logging.info("Converting text to lowercase...")
        self.df[self.text_column] = self.df[self.text_column].str.lower()

    def remove_html_tags(self):
        logging.info("Removing HTML tags...")
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: re.sub(r'<.*?>', '', str(x)))

    def remove_urls(self):
        logging.info("Removing URLs...")
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', str(x)))

    def normalize_text(self):
        logging.info("Expanding contractions...")
        self.df[self.text_column] = self.df[self.text_column].apply(lambda x: self.contraction_expander.expand_texts([x], precise=True)[0])

    def remove_punctuation(self):
        logging.info("Removing punctuation...")
        self.df[self.text_column] = self.df[self.text_column].str.replace(r'[^\w\s]', '', regex=True)

    def remove_stopwords(self):
        logging.info("Removing stopwords...")
        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda x: " ".join([word for word in str(x).split() if word not in self.stop_words])
        )

    def lemmatize_text(self):
        logging.info("Lemmatizing text...")
        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda x: " ".join([self.lemmatizer.lemmatize(token) for token in self.tokenizer.tokenize(x)])
        )

    def final_cleanup(self):
        logging.info("Performing final cleanups...")
        def clean_text(text):
            text = text.replace('\n', ' ').replace('\r', '')
            text = text.replace('"', '').replace("'", '')
            text = re.sub(r'\s{2,}', ' ', text)
            return text.strip()

        self.df[self.text_column] = self.df[self.text_column].apply(clean_text)

    def preprocess_all(self):
        self.extract_hashtags()
        self.remove_usernames()
        self.lowercase_text()
        self.remove_html_tags()
        self.remove_urls()
        self.normalize_text()
        self.remove_punctuation()
        self.remove_stopwords()
        self.lemmatize_text()
        self.final_cleanup()
        logging.info("All preprocessing steps completed.")
        return self.df

def main(input_path: str, output_path: str, column: str):
    logging.info("Reading input CSV file...")
    df = pd.read_csv(Config.DATA_PATH)
    processor = Preprocessor(df, column)
    cleaned_df = processor.preprocess_all()
    cleaned_df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--column', type=str, default='text', help='Name of the column to preprocess')
    args = parser.parse_args()
    main(args.input, args.output, args.column)
