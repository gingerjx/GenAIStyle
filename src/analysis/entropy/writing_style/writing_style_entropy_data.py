from dataclasses import dataclass, field
from typing import Dict, List
from pandas._libs.interval import Interval

from src.analysis.metrics.common.metrics_data import MetricData

@dataclass
class FeatureData:
    name: str
    value: float
    collection_name: str
    author_name: str
    source_name: str

@dataclass
class BinData:
    interval: Interval
    index: int
    count: int
    probability: float
    features: List[FeatureData]

@dataclass
class FeatureDistributionData:
    min_value: float
    max_value: float
    size: int
    bins: List[BinData]

@dataclass
class CollectionEntropyAverageData:
    average: float = 0.0
    average_uncertainty: float = 0.0
    std: float = 0.0

@dataclass
class ChunkFeaturesEntropyData:
    features_entropy: Dict[str, float] = field(default_factory=dict)

@dataclass 
class ChunkSequenceEntropyData:
    entropy: float
    match_lengths: List[float]

@dataclass
class CollectionEntropyData:
    chunks_features_entropies: Dict[str, ChunkFeaturesEntropyData] = field(default_factory=dict)
    chunks_sequence_entropy: Dict[str, ChunkSequenceEntropyData] = field(default_factory=dict)
    average_data: Dict[str, CollectionEntropyAverageData] = field(default_factory=dict)
    average_chunk_id: str = ""

@dataclass
class EntropyResults:
    collection_names: List[str]

    features_distributions: Dict[str, FeatureDistributionData] = field(default_factory=dict)  # [feature_name]
    collections_entropies: Dict[str, CollectionEntropyData] = field(default_factory=dict)     # [collection_name]

    def __post_init__(self) -> None:
        for collection_name in self.collection_names:   
            self.collections_entropies[collection_name] = CollectionEntropyData()