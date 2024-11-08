from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

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

    def info(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["author", "collection", "num_of_syllabes", "num_of_complex_words", "num_of_sentences", "num_of_words"])
        for author_name in self.author_names:
            for collection_name in self.collection_names:
                data_series = self.full[author_name][collection_name].info()
                data_series["author"] = author_name
                data_series["collection"] = collection_name
                df = pd.concat([df, data_series.to_frame().T],ignore_index=True)
        return df
    
    def get_all_chunks_preprocessing_data(self) -> List[PreprocessingData]:
        return [chunk_data for author_name in self.author_names for collection_name in self.collection_names for chunk_data in self.chunks[author_name][collection_name]]
