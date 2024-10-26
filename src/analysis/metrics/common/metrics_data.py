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

@dataclass
class MetricsAnalysisResults:

    author_names: List[str]
    collection_names: List[str]
    full_author_collection: Dict[str, Dict[str, MetricData]] = field(init=False, default_factory=dict)              # [author][collection]
    chunks_author_collection: Dict[str, Dict[str, List[MetricData]]] = field(init=False, default_factory=dict)      # [author][collection][chunk_id]
    full_collection_author: Dict[str, Dict[str, MetricData]] = field(init=False, default_factory=dict)              # [collection][author]
    chunks_collection_author: Dict[str, Dict[str, List[MetricData]]] = field(init=False, default_factory=dict)      # [collection][author][chunk_id]
    global_top_function_words: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        for author_name in self.author_names:   
            self.full_author_collection[author_name] = {}
            self.chunks_author_collection[author_name] = {}
            for collection_name in self.collection_names:
                self.chunks_author_collection[author_name].update({
                    collection_name: []
                })
        
        for collection_name in self.collection_names: 
            self.full_collection_author[collection_name] = {}
            self.chunks_collection_author[collection_name] = {}
            for author_name in self.author_names: 
                self.chunks_collection_author[collection_name].update({
                    author_name: []
                })

    def get_all_full_metrics(self) -> List[MetricData]:
        return [self.full_author_collection[author_name][collection_name] for author_name in self.author_names for collection_name in self.collection_names]
    
    def get_all_chunks_metrics(self) -> List[MetricData]:
        return [metric for author_name in self.author_names for collection_name in self.collection_names for metric in self.chunks_author_collection[author_name][collection_name]]