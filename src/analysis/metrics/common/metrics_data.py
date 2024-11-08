from abc import abstractmethod
import abc
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricData:

    chunk_id: str
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

    def __hash__(self):
        return hash((
            self.source_name,
            self.author_name,
            self.collection_name,
            self.unique_word_count,
            self.average_word_length,
            self.average_sentence_length,
            frozenset(self.sorted_function_words.items()),
            frozenset(self.punctuation_frequency.items()),
            self.average_syllables_per_word,
            self.flesch_reading_ease,
            self.flesch_kincaid_grade_level,
            self.gunning_fog_index,
            self.yules_characteristic_k,
            self.herdans_c,
            self.maas,
            self.simpsons_index
        ))

    def __eq__(self, other):
        if not isinstance(other, MetricData):
            return False
        return (
            self.source_name == other.source_name and
            self.author_name == other.author_name and
            self.collection_name == other.collection_name and
            self.unique_word_count == other.unique_word_count and
            self.average_word_length == other.average_word_length and
            self.average_sentence_length == other.average_sentence_length and
            self.sorted_function_words == other.sorted_function_words and
            self.punctuation_frequency == other.punctuation_frequency and
            self.average_syllables_per_word == other.average_syllables_per_word and
            self.flesch_reading_ease == other.flesch_reading_ease and
            self.flesch_kincaid_grade_level == other.flesch_kincaid_grade_level and
            self.gunning_fog_index == other.gunning_fog_index and
            self.yules_characteristic_k == other.yules_characteristic_k and
            self.herdans_c == other.herdans_c and
            self.maas == other.maas and
            self.simpsons_index == other.simpsons_index
        )
    
@dataclass
class MetricsAnalysisResults(abc.ABC):

    @abstractmethod
    def get_all_full_metrics(self) -> List[MetricData]:
        pass
    
    @abstractmethod
    def get_all_chunks_metrics(self) -> List[MetricData]:
        pass