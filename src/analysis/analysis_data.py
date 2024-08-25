from typing import Dict


class AnalysisData():

    def __init__(self, 
                author_name: str, 
                collection_name: str, 
                unique_word_count: int,
                average_word_length: float,
                average_sentence_length: float,
                top_10_function_words: Dict[str, int],
                punctuation_frequency: Dict[str, int],
                average_syllables_per_word: float,
                flesch_reading_ease: float,
                flesch_kincaid_grade_level: float,
                gunning_fog_index: float) -> None:
        self.author_name = author_name
        self.collection_name = collection_name
        self.unique_word_count = unique_word_count
        self.average_word_length = average_word_length
        self.average_sentence_length = average_sentence_length
        self.top_10_function_words = top_10_function_words
        self.punctuation_frequency = punctuation_frequency
        self.average_syllables_per_word = average_syllables_per_word
        self.flesch_reading_ease = flesch_reading_ease
        self.flesch_kincaid_grade_level = flesch_kincaid_grade_level
        self.gunning_fog_index = gunning_fog_index