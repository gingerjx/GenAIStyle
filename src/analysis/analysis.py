from typing import Dict, List
import nltk
from functionwords import FunctionWords
from src import Author

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from src.analysis.alanysis_data import AnalysisData

class Analysis():

    def __init__(self, size: int = None):
        self.size = size

    def analyze(self, authors: List[Author]) -> Dict[str, List[AnalysisData]]:
        """Analyze the authors and their collections"""
        data = {}

        for author in authors:
            for collection in author.collections:
                merged_text = collection.get_merged_text()
                analysis_data = AnalysisData(author.name, collection.name, **self._analyze(merged_text))
                model_name = collection.name
                if model_name not in data:
                    data[model_name] = []

                data[model_name].append(analysis_data)
        return data

    def _analyze(self, text: str) -> dict:
        """Analyze the text and return the word_counts, unique_word_counts, average_word_lengths and average_sentence_lengths"""
        data = {}
        words = text.split()
        data["word_count"] = len(words)
        if self.size: 
            words = words[:self.size]
            text = " ".join(words)

        data["unique_word_count"] = len(set(words))
        data["average_word_length"] = sum(len(word) for word in words) / len(words)

        sentences = sent_tokenize(text)
        data["average_sentence_length"] = len(words) / len(sentences)

        data["top_10_function_words"] = self._get_top_function_words(text, n=10)

        return data
    
    def _get_top_function_words(self, text: str, n: int) -> Dict[str, int]:
        """Get the top n function words from the text"""
        fw = FunctionWords(function_words_list="english")
        fw_frequency = dict(zip(fw.get_feature_names(), fw.transform(text)))
        sorted_fw_frequency = sorted(fw_frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_fw_frequency[:n])