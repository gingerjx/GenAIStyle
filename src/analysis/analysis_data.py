from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class MetricData:
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

@dataclass
class PCAData:
    data: dict = None
    results: pd.DataFrame = None
    pc_variance: List[float] = None
    top_10_features: Dict[str, List[str]] = None

@dataclass
class AnalysisData:
    author_names: List[str]
    collection_names: List[str]
    percentage_of_removed_text: float
    cross_top_function_words_names: List[str] = field(default_factory=list)
    collection_metrics: Dict[str, List] = field(init=False, default_factory=dict)
    author_metrics: Dict[str, List] = field(init=False, default_factory=dict)
    pca: PCAData = field(default_factory=PCAData)

    def __post_init__(self):
        for collection_name in self.collection_names:
            self.collection_metrics[collection_name] = []
        for author_name in self.author_names:
            self.author_metrics[author_name] = []