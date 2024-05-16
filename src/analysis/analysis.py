from typing import Dict, List
import nltk
from functionwords import FunctionWords
from src import Author

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from src.analysis.alanysis_data import AnalysisData

class Analysis():
    EXLUDED_FUNCTION_WORDS = [
        "the"
    ]

    def __init__(self, size: int = None):
        self.size = size

    def analyze(self, authors: List[Author]) -> Dict[str, List[AnalysisData]]:
        data = {}  # Initialize an empty dictionary to store the results

        for author in authors:
            for collection in author.collections:
                merged_text = collection.get_merged_text()
                analysis_data = AnalysisData(author.name, collection.name, **self._analyze(merged_text))

                # Use collection name (model) as the key
                model_name = collection.name
                if model_name not in data:
                    data[model_name] = []

                data[model_name].append(analysis_data)
        return data

    def _analyze(self, text: str) -> dict:
        data = {}
        words = text.split()
        if self.size: 
            words = words[:self.size]
            text = " ".join(words)

        data["word_count"] = len(words)
        data["unique_word_count"] = len(set(words))
        data["average_word_length"] = sum(len(word) for word in words) / len(words)

        sentences = sent_tokenize(text)
        data["average_sentence_length"] = len(words) / len(sentences)

        data["top_10_function_words"] = self._get_top_function_words(text, n=10)

        return data
    
    def _get_top_function_words(self, text: str, n: int) -> Dict[str, int]:
        fw = FunctionWords(function_words_list="english")
        fw_frequency = dict(zip(fw.get_feature_names(), fw.transform(text)))
        fw_frequency_filtered = self._exlude_function_words(fw_frequency)
        sorted_fw_frequency_filtered = sorted(fw_frequency_filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_fw_frequency_filtered[:n])
    
    def _exlude_function_words(self, fw_frequency: dict) -> dict:
        for word in self.EXLUDED_FUNCTION_WORDS:
            fw_frequency.pop(word, None)
        return fw_frequency