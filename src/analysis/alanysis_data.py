import json
from typing import Any, Dict


class AnalysisData():

    def __init__(self, 
                 author_name: str, 
                 collection_name: str, 
                 word_count: int,
                 unique_word_count: int,
                 average_word_length: float,
                 average_sentence_length: float,
                 top_10_function_words: Dict[str, int]):
        self.author_name = author_name
        self.collection_name = collection_name
        self.word_count = word_count
        self.unique_word_count = unique_word_count
        self.average_word_length = average_word_length
        self.average_sentence_length = average_sentence_length
        self.top_10_function_words = top_10_function_words