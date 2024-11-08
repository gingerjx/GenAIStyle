from dataclasses import dataclass
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
class ChunkFeatureEntropyData:
    total_entropy: float
    features_entropy: Dict[str, float]

@dataclass 
class ChunkSequenceEntropyData:
    total_entropy: float
    match_lengths: List[float]

@dataclass
class EntropyResults:
    distributions: Dict[str, FeatureDistributionData]                # [feature_name]
    all_chunks_features_entropy: Dict[str, ChunkFeatureEntropyData]  # [chunk_id]
    all_chunks_sequence_entropy: Dict[str, ChunkSequenceEntropyData] # [chunk_id]