from typing import Dict, List
import nltk
import json
from functionwords import FunctionWords
from src import Author
from src.analysis.preprocessing import Preprocessing
from src.file_utils import FileUtils
from src.settings import Settings
from src.models.collection import Collection
import jsonpickle
from string import punctuation
from collections import Counter

nltk.download('punkt')

from nltk.tokenize import sent_tokenize

class Analysis():

    class AnalysisData():

        def __init__(self, 
                    author_name: str, 
                    collection_name: str, 
                    unique_word_count: int,
                    average_word_length: float,
                    average_sentence_length: float,
                    top_10_function_words: Dict[str, int],
                    punctuation_frequency: Dict[str, int],
                    average_syllables_per_word: float) -> None:
            self.author_name = author_name
            self.collection_name = collection_name
            self.unique_word_count = unique_word_count
            self.average_word_length = average_word_length
            self.average_sentence_length = average_sentence_length
            self.top_10_function_words = top_10_function_words
            self.punctuation_frequency = punctuation_frequency
            self.average_syllables_per_word = average_syllables_per_word

    def __init__(self, 
                 settings: Settings, 
                 authors: List[Author], 
                 preprocessing_data: Preprocessing.Data
            ) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.authors = authors
        self.preprocessing_data = preprocessing_data

    def text_length(self) -> int:
        raw_text_length = 0
        cleaned_text_length = 0
        for author in self.authors:
            raw_text_length += self._get_text_length(author.raw_collections)
            cleaned_text_length += self._get_text_length(author.cleaned_collections)
        return raw_text_length, cleaned_text_length
    
    def get_analysis(self, authors: List[Author], read_from_file=False) -> Dict[str, List[AnalysisData]]:    
        if read_from_file:
            return FileUtils.read_analysis_data(self.paths.analysis_filepath)
        return self.analyze(authors)
    
    def analyze(self, authors: List[Author]) -> Dict[str, List[AnalysisData]]:
        """Analyze the authors and their collections"""
        data = {}

        for author in authors:
            for collection in author.cleaned_collections:
                model_name = collection.name
                if model_name not in data:
                    data[model_name] = []

                collection_metrics = self._analyze(self.preprocessing_data[author][collection])
                analysis_data = Analysis.AnalysisData(
                    author_name=author.name, 
                    collection_name=collection.name, 
                    **collection_metrics
                )

                data[model_name].append(analysis_data)

        self._save_analysis_data(data)
        return data

    def _get_text_length(self, collections: List[Collection]) -> int:
        """Get the total number of words in the collections"""
        all_text = " ".join([collection.get_merged_text() for collection in collections])
        return len(all_text)
    
    def _save_analysis_data(self, data: Dict[str, List[AnalysisData]]) -> None:
        """Save the analysis data to a file"""
        self.paths.analysis_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths.analysis_filepath, 'w', encoding='utf-8') as f:
            json_data = jsonpickle.encode(data)
            json.dump(json_data, f, indent=4)
    
    def _analyze(self, preprocessing_data: Preprocessing.Data) -> dict:
        """Analyze the sample of words and return the unique_word_counts, average_word_lengths and average_sentence_lengths"""
        data = {}
        data["unique_word_count"] = self._get_unique_word_count(
            words=preprocessing_data.words
        )
        data["average_word_length"] = self._get_average_word_length(
            words=preprocessing_data.words
        )
        data["average_sentence_length"] = self._get_average_sentence_length(
            text=preprocessing_data.text, 
            words=preprocessing_data.words
        )
        data["top_10_function_words"] = self._get_top_function_words(
            text=preprocessing_data.text
        )
        data["punctuation_frequency"] = self._get_punctuation_frequency(
            text=preprocessing_data.text
        )
        data["average_syllables_per_word"] = self._average_syllables_per_word(
            words=preprocessing_data.words, 
            syllables_count=preprocessing_data.syllables_count
        )
        return data

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
    
    def _get_punctuation_frequency(self, text: str) -> Dict[str, int]:
        """Get the punctuation frequency from the text"""
        counts = Counter(text)
        result = {k:v for k, v in counts.items() if k in punctuation}
        missing_punctiation = set(punctuation) - set(result.keys())
        result.update({k:0 for k in missing_punctiation})
        return result
    
    def _average_syllables_per_word(self, words: List[str], syllables_count) -> float:
        """Get the average syllables per word"""
        return syllables_count / len(words)