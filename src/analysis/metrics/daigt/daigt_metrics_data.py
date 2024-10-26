from dataclasses import dataclass, field
from typing import Dict, List

from src.analysis.metrics.common.metrics_data import MetricData


@dataclass
class DaigtMetricsAnalysisResults:

    collection_names: List[str]
    full_collection: Dict[str, MetricData] = field(init=False, default_factory=dict)              # [collection]
    chunks_collection: Dict[str, List[MetricData]] = field(init=False, default_factory=dict)      # [collection][chunk_id]
    global_top_function_words: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:        
        for collection_name in self.collection_names: 
            self.chunks_collection[collection_name] = []

    def get_all_full_metrics(self) -> List[MetricData]:
        return [self.full_collection[collection_name] for collection_name in self.collection_names]

    def get_all_chunks_metrics(self) -> List[MetricData]:
        return [metric for collection_name in self.collection_names for metric in self.chunks_collection[collection_name]]