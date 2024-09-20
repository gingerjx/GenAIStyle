from typing import Dict, List
import nltk
import json
from functionwords import FunctionWords
from src.analysis.analysis_data import AnalysisData, MetricData, Metadata
from src.analysis.pca_analysis import PCAAnalysis
from src.analysis.preprocessing_data import PreprocessingData
from src.file_utils import FileUtils
from src.models.author import Author
from src.settings import Settings
from src.models.collection import Collection
import jsonpickle
from string import punctuation
from collections import Counter
import pandas as pd
from dataclasses import fields
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math

class Analysis():

    def __init__(self, 
                 settings: Settings, 
                 authors: List[Author], 
                 preprocessing_data: PreprocessingData,
                 read_from_file: bool = False
            ) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.authors = authors
        self.preprocessing_data = preprocessing_data
        self.read_from_file = read_from_file
    
    def get_analysis(self, authors: List[Author]) -> AnalysisData:    
        if self.read_from_file:
            return FileUtils.read_analysis_data(self.paths.analysis_filepath)
        return self.analyze(authors)
    
    def analyze(self, authors: List[Author]) -> AnalysisData:
        """Analyze the authors and their collections"""
        analysis_data = AnalysisData(
            author_names=[author.name for author in authors],
            collection_names=[collection.name for collection in authors[0].cleaned_collections],
            metadata=Metadata(percentage_of_removed_text=self._get_percentage_of_removed_text())
        )

        for author in authors:
            for collection in author.cleaned_collections:
                model_name = collection.name
                collection_metrics = self._analyze(self.preprocessing_data[author][collection])
                metrics = MetricData(
                    author_name=author.name, 
                    collection_name=collection.name, 
                    **collection_metrics
                )
                analysis_data.collection_metrics[model_name].append(metrics)

        analysis_data.metadata.cross_top_function_words_names = self._get_cross_top_function_words_names(analysis_data)
        analysis_data.pca = PCAAnalysis.get_analysis(analysis_data)
        self._set_author_metrics(analysis_data)
        
        self._save_analysis_data(analysis_data)

        return analysis_data

    def _get_cross_top_function_words_names(self, analysis_data: AnalysisData) -> List[str]:
        """Get the top n function words from all the collections"""
        cross_top_function_words_names = []
        for collection_name in analysis_data.collection_names:
            for metrics in analysis_data.collection_metrics[collection_name]:
                cross_top_function_words_names.extend(list(metrics.sorted_function_words.keys())[:self.configuration.top_n_function_words])
        return list(set(cross_top_function_words_names))
   
    def _set_author_metrics(self, analysis_data: AnalysisData) -> None:
        """Set the author metrics"""
        for collection_name, metrics in analysis_data.collection_metrics.items():
             for metric in metrics:
                analysis_data.author_metrics[metric.author_name].append(metric)

    def _get_percentage_of_removed_text(self) -> float:
        raw_text_length = 0
        cleaned_text_length = 0
        for author in self.authors:
            raw_text_length += self._get_text_length(author.raw_collections)
            cleaned_text_length += self._get_text_length(author.cleaned_collections)
        return 100 * (raw_text_length - cleaned_text_length) / raw_text_length
    
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
    
    def _analyze(self, preprocessing_data: PreprocessingData) -> dict:
        """Analyze the sample of words and return the unique_word_counts, average_word_lengths and average_sentence_lengths"""
        data = {}
        
        data["unique_word_count"] = self._get_unique_word_count(
            words=preprocessing_data.words
        )
        data["average_word_length"] = self._get_average_word_length(
            words=preprocessing_data.words,
            num_of_words=preprocessing_data.num_of_words
        )
        data["average_sentence_length"] = self._get_average_sentence_length(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences
        )
        data["sorted_function_words"] = self._get_sorted_function_words(
            words=preprocessing_data.words
        )
        data["punctuation_frequency"] = self._get_punctuation_frequency(
            text=preprocessing_data.text
        )
        data["average_syllables_per_word"] = self._average_syllables_per_word(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_syllabes=preprocessing_data.num_of_syllabes
        )
        data["flesch_reading_ease"] = self._get_flesch_reading_ease(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences, 
            num_of_syllabes=preprocessing_data.num_of_syllabes
        )
        data["flesch_kincaid_grade_level"] = self._get_flesch_kincaid_grade_level(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences, 
            num_of_syllabes=preprocessing_data.num_of_syllabes
        )
        data["gunning_fog_index"] = self._gunning_fog_index(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences, 
            num_of_complex_words=preprocessing_data.num_of_complex_words
        )
        data["yules_characteristic_k"] = self._yules_characteristic_k(
            words=preprocessing_data.words
        )
        data["herdans_c"] = self._herdans_c(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_unique_words=data["unique_word_count"]
        )
        data["maas"] = self._maas(
            num_of_words=preprocessing_data.num_of_words, 
            vocabulary_size=data["unique_word_count"]
        )
        data["simpsons_index"] = self._simpsons_index(
            words=preprocessing_data.words
        )
        
        return data
 
    # SINGLE METRICS

    def _get_unique_word_count(self, words: List[str]) -> int:
        """Get the unique word count from the text"""
        return len(set(words))

    def _get_average_word_length(self, words: List[str], num_of_words: int) -> float:
        """Get the average word length from the text"""
        return sum(len(word) for word in words) / num_of_words
    
    def _get_average_sentence_length(self, num_of_words: int, num_of_sentences: int) -> float:
        """Get the average word length from the text"""
        return num_of_words / num_of_sentences
    
    def _get_sorted_function_words(self, words: List[str]) -> Dict[str, int]:  
        """Get the function words from the text"""
        fw = FunctionWords(function_words_list="english")
        fw_frequency = {}
        for word in set(words):
            if word in fw.get_feature_names(): # TODO: Should we add `'s`, `'d`, `'ll` etc.?
                fw_frequency[word] = words.count(word)
        sorted_fw_frequency = sorted(fw_frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_fw_frequency)
        
    def _get_punctuation_frequency(self, text: str) -> Dict[str, int]:
        """Get the punctuation frequency from the text"""
        counts = Counter(text)
        result = {k:v for k, v in counts.items() if k in punctuation}
        missing_punctiation = set(punctuation) - set(result.keys())
        result.update({k:0 for k in missing_punctiation})
        return result
    
    def _average_syllables_per_word(self, num_of_words: int, num_of_syllabes: int) -> float:
        """Get the average syllables per word"""
        return num_of_syllabes / num_of_words
    
    def _get_flesch_reading_ease(self, num_of_words: int, num_of_sentences: List[str], num_of_syllabes: int) -> float:
        """Get the Flesch Reading Ease score"""
        return 206.835 - 1.015 * (num_of_words / num_of_sentences) - 84.6 * (num_of_syllabes / num_of_words)
    
    def _get_flesch_kincaid_grade_level(self, num_of_words: int, num_of_sentences: List[str], num_of_syllabes: int) -> float:
        """Get the Flesch-Kincaid Grade Level score"""
        return 0.39 * (num_of_words / num_of_sentences) + 11.8 * (num_of_syllabes / num_of_words) - 15.59
    
    def _gunning_fog_index(self, num_of_words: int, num_of_sentences: int, num_of_complex_words: int) -> float:
        """Get the Gunning Fog Index score"""
        return 0.4 * ((num_of_words / num_of_sentences) + 100 * (num_of_complex_words / num_of_words))
    
    def _yules_characteristic_k(self, words: List[str]) -> float:
        """Get the Yule's Characteristic K score"""
        word_freqs = nltk.FreqDist(words)
        N = len(words)
        V = len(set(words))
        c = N/V
        K = 10**4/(N**2) * sum([(freq - c)**2 for freq in word_freqs.values()])
        return K
    
    def _herdans_c(self, num_of_words: int, num_of_unique_words: int) -> float:
        """Get the Herdan's C score"""
        return num_of_unique_words / num_of_words
    
    def _maas(self, num_of_words: int, vocabulary_size: int) -> float:
        """Get the Maas score"""
        return (math.log(num_of_words) - math.log(vocabulary_size)) / math.log(num_of_words**2)
    
    def _simpsons_index(self, words: List[str]) -> float:
        """Get the Simpson's Index score"""
        word_freqs = nltk.FreqDist(words)
        N = len(words)
        return 1 - sum([(freq/N)**2 for freq in word_freqs.values()])