from collections import Counter
import math
from string import punctuation
from typing import Dict, List
from src.analysis.preprocessing_data import PreprocessingData
import nltk
from functionwords import FunctionWords

class MetricsAnalysis:

    @staticmethod
    def _analyze(preprocessing_data: PreprocessingData) -> dict:
        """Analyze the sample of words and return the unique_word_counts, average_word_lengths and average_sentence_lengths"""
        data = {"source_name": preprocessing_data.source_name}
        
        data["unique_word_count"] = MetricsAnalysis._get_unique_word_count(
            words=preprocessing_data.words
        )
        data["average_word_length"] = MetricsAnalysis._get_average_word_length(
            words=preprocessing_data.words,
            num_of_words=preprocessing_data.num_of_words
        )
        data["average_sentence_length"] = MetricsAnalysis._get_average_sentence_length(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences
        )
        data["sorted_function_words"] = MetricsAnalysis._get_sorted_function_words(
            words=preprocessing_data.words
        )
        data["punctuation_frequency"] = MetricsAnalysis._get_punctuation_frequency(
            text=preprocessing_data.text
        )
        data["average_syllables_per_word"] = MetricsAnalysis._average_syllables_per_word(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_syllabes=preprocessing_data.num_of_syllabes
        )
        data["flesch_reading_ease"] = MetricsAnalysis._get_flesch_reading_ease(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences, 
            num_of_syllabes=preprocessing_data.num_of_syllabes
        )
        data["flesch_kincaid_grade_level"] = MetricsAnalysis._get_flesch_kincaid_grade_level(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences, 
            num_of_syllabes=preprocessing_data.num_of_syllabes
        )
        data["gunning_fog_index"] = MetricsAnalysis._gunning_fog_index(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_sentences=preprocessing_data.num_of_sentences, 
            num_of_complex_words=preprocessing_data.num_of_complex_words
        )
        data["yules_characteristic_k"] = MetricsAnalysis._yules_characteristic_k(
            words=preprocessing_data.words
        )
        data["herdans_c"] = MetricsAnalysis._herdans_c(
            num_of_words=preprocessing_data.num_of_words, 
            num_of_unique_words=data["unique_word_count"]
        )
        data["maas"] = MetricsAnalysis._maas(
            num_of_words=preprocessing_data.num_of_words, 
            vocabulary_size=data["unique_word_count"]
        )
        data["simpsons_index"] = MetricsAnalysis._simpsons_index(
            words=preprocessing_data.words
        )
        
        return data
    
    def _get_unique_word_count(words: List[str]) -> int:
        """Get the unique word count from the text"""
        return len(set(words))

    def _get_average_word_length(words: List[str], num_of_words: int) -> float:
        """Get the average word length from the text"""
        return sum(len(word) for word in words) / num_of_words
    
    def _get_average_sentence_length(num_of_words: int, num_of_sentences: int) -> float:
        """Get the average word length from the text"""
        return num_of_words / num_of_sentences
    
    def _get_sorted_function_words(words: List[str]) -> Dict[str, int]:  
        """Get the function words from the text"""
        fw = FunctionWords(function_words_list="english")
        fw_frequency = {}
        for word in set(words):
            if word in fw.get_feature_names(): # TODO: Should we add `'s`, `'d`, `'ll` etc.?
                fw_frequency[word] = words.count(word)
        sorted_fw_frequency = sorted(fw_frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_fw_frequency)
        
    def _get_punctuation_frequency(text: str) -> Dict[str, int]:
        """Get the punctuation frequency from the text"""
        counts = Counter(text)
        result = {k:v for k, v in counts.items() if k in punctuation}
        missing_punctiation = set(punctuation) - set(result.keys())
        result.update({k:0 for k in missing_punctiation})
        return result
    
    def _average_syllables_per_word(num_of_words: int, num_of_syllabes: int) -> float:
        """Get the average syllables per word"""
        return num_of_syllabes / num_of_words
    
    def _get_flesch_reading_ease(num_of_words: int, num_of_sentences: List[str], num_of_syllabes: int) -> float:
        """Get the Flesch Reading Ease score"""
        return 206.835 - 1.015 * (num_of_words / num_of_sentences) - 84.6 * (num_of_syllabes / num_of_words)
    
    def _get_flesch_kincaid_grade_level(num_of_words: int, num_of_sentences: List[str], num_of_syllabes: int) -> float:
        """Get the Flesch-Kincaid Grade Level score"""
        return 0.39 * (num_of_words / num_of_sentences) + 11.8 * (num_of_syllabes / num_of_words) - 15.59
    
    def _gunning_fog_index(num_of_words: int, num_of_sentences: int, num_of_complex_words: int) -> float:
        """Get the Gunning Fog Index score"""
        return 0.4 * ((num_of_words / num_of_sentences) + 100 * (num_of_complex_words / num_of_words))
    
    def _yules_characteristic_k(words: List[str]) -> float:
        """Get the Yule's Characteristic K score"""
        word_freqs = nltk.FreqDist(words)
        N = len(words)
        V = len(set(words))
        c = N/V
        K = 10**4/(N**2) * sum([(freq - c)**2 for freq in word_freqs.values()])
        return K
    
    def _herdans_c(num_of_words: int, num_of_unique_words: int) -> float:
        """Get the Herdan's C score"""
        return num_of_unique_words / num_of_words
    
    def _maas(num_of_words: int, vocabulary_size: int) -> float:
        """Get the Maas score"""
        return (math.log(num_of_words) - math.log(vocabulary_size)) / math.log(num_of_words**2)
    
    def _simpsons_index(words: List[str]) -> float:
        """Get the Simpson's Index score"""
        word_freqs = nltk.FreqDist(words)
        N = len(words)
        return 1 - sum([(freq/N)**2 for freq in word_freqs.values()])