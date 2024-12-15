import re
from typing import List
import nltk
import pandas as pd
from src.settings import Settings
import os
import json

class Dataset:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.corpus = ""

    def load(self):
        pass
    
    def get_words(self) -> List[str]:
        split = nltk.word_tokenize(self.corpus)
        words = [word for word in split if re.search(r'[a-zA-Z]', word)]
        return words

class TwitterDataset(Dataset):
    
    # https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset
    def load(self) -> "TwitterDataset":
        df = pd.read_csv(self.settings.paths.twitter_raw_dataset_filepath)
        df['clean_text'] = df['clean_text'].astype(str)
        self.corpus = df['clean_text'].str.cat(sep=' ')
        return self

class RedditDataset(Dataset):
        
    # https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset
    def load(self) -> "RedditDataset":
        df = pd.read_csv(self.settings.paths.reddit_raw_dataset_filepath)
        df['clean_comment'] = df['clean_comment'].astype(str)
        self.corpus = df['clean_comment'].str.cat(sep=' ')
        return self

class NewsDataset(Dataset):
    
    # https://www.kaggle.com/datasets/everydaycodings/global-news-dataset
    def load(self) -> "NewsDataset":
        df = pd.read_csv(self.settings.paths.news_raw_dataset_filepath)
        df['content'] = df['content'].astype(str)
        self.corpus = df['content'].str.cat(sep=' ')
        return self

class LegalDataset(Dataset):
    
    # https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta
    def load(self) -> "LegalDataset":
        path = self.settings.paths.legal_raw_dir
        files = os.listdir(path)

        for file in files:
            with open(os.path.join(path, file), 'r', encoding="utf-8") as f:
                self.corpus += f.read() + " "

        return self
    
class DaigtDataset(Dataset):

    def load(self) -> "DaigtDataset":
        df = pd.read_csv(self.settings.paths.daigt_raw_dataset_filepath)
        df['text'] = df['text'].astype(str)
        self.corpus = df['text'].str.cat(sep=' ')
        return self

class WritingStyleDataset(Dataset):

    def load(self) -> "WritingStyleDataset":
        path = self.settings.paths.ws_raw_books_dir
        selected_books = pd.read_csv(self.settings.paths.ws_selected_books_csv_filepath)

        books_path = selected_books["author"] + "___" + selected_books["book"] + ".txt"
        for book_path in books_path:
            with open(os.path.join(path, book_path), 'r', encoding="utf-8") as f:
                self.corpus += f.read() + " "

        return self

