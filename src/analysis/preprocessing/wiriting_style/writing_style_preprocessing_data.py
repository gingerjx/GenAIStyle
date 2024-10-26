from dataclasses import dataclass, field
from typing import Dict, List

from src.analysis.preprocessing.common.preprocessing_data import PreprocessingData


@dataclass
class WritingStylePreprocessingResults:
    
    author_names: List[str]
    collection_names: List[str]
    full: Dict[str, Dict[str, PreprocessingData]] = field(init=False, default_factory=dict)              # [author][collection]
    chunks: Dict[str, Dict[str, List[PreprocessingData]]] = field(init=False, default_factory=dict)      # [author][collection][chunk_id]

    def __post_init__(self) -> None:
        for author_name in self.author_names:   
            self.chunks[author_name] = {}
            self.full[author_name] = {}
            for collection_name in self.collection_names:
                self.full[author_name][collection_name] = PreprocessingData()
                self.chunks[author_name].update({
                    collection_name: []
                })