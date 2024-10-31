from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.analysis.preprocessing.common.preprocessing_data import PreprocessingData


@dataclass
class DaigtPreprocessingResults:
    
    collection_names: List[str]
    full: Dict[str, PreprocessingData] = field(init=False, default_factory=dict)              # [collection]
    chunks: Dict[str, List[PreprocessingData]] = field(init=False, default_factory=dict)      # [collection][chunk_id]

    def info(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["collection", "num_of_syllabes", "num_of_complex_words", "num_of_sentences", "num_of_words"])
        for collection_name in self.collection_names:
            data_series = self.full[collection_name].info()
            data_series["collection"] = collection_name
            df = pd.concat([df, data_series.to_frame().T],ignore_index=True)
        return df