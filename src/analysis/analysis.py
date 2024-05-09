from typing import List
import nltk

from src import Author

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from src.analysis.alanysis_data import AnalysisData

class Analysis():

    def __init__(self, size: int = None):
        self.size = size

    def analyze(self, authors: List[Author]):
        data = {}
        for author in authors:
            data[author.name] = []
            for collection in author.collections:
                merged_text = collection.get_merged_text()
                data[author.name].append(AnalysisData(author.name, collection.name, **self._analyze(merged_text)))
        return data

    def _analyze(self, text: str) -> dict:
        data = {}
        words = text.split()
        if self.size: words = words[:self.size]

        data["word_count"] = len(words)
        data["unique_word_count"] = len(set(words))
        data["average_word_length"] = sum(len(word) for word in words) / len(words)
        sentences = sent_tokenize(text)
        data["average_sentence_length"] = len(sentences) / len(words)

        return data