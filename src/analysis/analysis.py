from typing import Dict, List
import nltk
from functionwords import FunctionWords
from src import Author
from src.settings import Settings
from src.models.collection import Collection

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from src.analysis.alanysis_data import AnalysisData

class Analysis():

    def __init__(self, settings: Settings, authors: List[Author]) -> None:
        self.configuration = settings.configuration
        self.authors = authors

    def get_count_words(self) -> int:
        raw_words_count = 0
        cleaned_words_count = 0
        for author in self.authors:
            raw_words_count += self._get_words_count(author.raw_collections)
            cleaned_words_count += self._get_words_count(author.cleaned_collections)
        return raw_words_count, cleaned_words_count
    
    def analyze(self, authors: List[Author]) -> Dict[str, List[AnalysisData]]:
        """Analyze the authors and their collections"""
        data = {}

        for author in authors:
            for collection in author.cleaned_collections:
                merged_text = collection.get_merged_text()
                analysis_data = AnalysisData(author.name, collection.name, **self._analyze(merged_text))
                model_name = collection.name
                if model_name not in data:
                    data[model_name] = []

                data[model_name].append(analysis_data)
        return data

    def _get_words_count(self, collections: List[Collection]) -> int:
        all_text = " ".join([collection.get_merged_text() for collection in collections])
        return len(all_text.split())
    
    def _analyze(self, text: str) -> dict:
        """Analyze the text and return the word_counts, unique_word_counts, average_word_lengths and average_sentence_lengths"""
        data = {}

        words = text.split()
        data["word_count"] = len(words)

        data.update(self._analyze_sample(
            text=self._merge_words(words), 
            words=self._get_words_sample(words)
            )
        )

        return data
    
    def _analyze_sample(self, text: str, words: List[str]) -> dict:
        data = {}
        data["unique_word_count"] = self._get_unique_word_count(words)
        data["average_word_length"] = self._get_average_word_length(words)
        data["average_sentence_length"] = self._get_average_sentence_length(text, words)
        data["top_10_function_words"] = self._get_top_function_words(text)
        return data

    def _get_words_sample(self, all_words: List[str]) -> List[str]:
        """Get a sample of words from the text"""
        return all_words[:self.configuration.analysis_size]

    def _merge_words(self, words: List[str]) -> str:
        """Merge the words into a single text"""
        return " ".join(words)
    
    def _get_unique_word_count(self, words: List[str]) -> int:
        """Get the unique word count from the text"""
        return len(set(words))

    def _get_average_word_length(self, words: List[str]) -> float:
        """Get the average word length from the text"""
        return sum(len(word) for word in words) / len(words)
    
    def _get_average_sentence_length(self, text: str, words: List[str]) -> float:
        """Get the average word length from the text"""
        try:
            sentences = sent_tokenize(text)
        except:
            pass
        return len(words) / len(sentences)
    
    def _get_top_function_words(self, text: str) -> Dict[str, int]:
        """Get the top n function words from the text"""
        fw = FunctionWords(function_words_list="english")
        fw_frequency = dict(zip(fw.get_feature_names(), fw.transform(text)))
        sorted_fw_frequency = sorted(fw_frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_fw_frequency[:self.configuration.n_top_function_words])