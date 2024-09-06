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
    top_10_function_words: Dict[str, int]
    punctuation_frequency: Dict[str, int]
    average_syllables_per_word: float
    flesch_reading_ease: float
    flesch_kincaid_grade_level: float
    gunning_fog_index: float

@dataclass
class AnalysisData:
    author_names: List[str]
    collection_names: List[str]
    percentage_of_removed_text: float
    all_top_function_words: List[str] = field(default_factory=list)
    collection_metrics: Dict[str, List] = field(init=False, default_factory=dict)
    pca_data: Optional[dict] = None
    pca_results: Optional[Dict[str, pd.DataFrame]] = None

    def __post_init__(self):
        for collection_name in self.collection_names:
            self.collection_metrics[collection_name] = []