from dataclasses import dataclass
from typing import Dict, List
from pandas._libs.interval import Interval

@dataclass
class EntropyFeatureData:
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
    features: List[EntropyFeatureData]

@dataclass
class FeatureDistributionData:
    min_value: float
    max_value: float
    size: int
    bins: List[BinData]

@dataclass
class EntropyData:
    distributions: Dict[str, FeatureDistributionData]