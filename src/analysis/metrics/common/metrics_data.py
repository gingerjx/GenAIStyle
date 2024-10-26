from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricData:
    source_name: str
    author_name: str
    collection_name: str
    unique_word_count: int
    average_word_length: float
    average_sentence_length: float
    sorted_function_words: Dict[str, int]
    punctuation_frequency: Dict[str, int]
    average_syllables_per_word: float
    flesch_reading_ease: float
    flesch_kincaid_grade_level: float
    gunning_fog_index: float
    yules_characteristic_k: float
    herdans_c: float
    maas: float
    simpsons_index: float