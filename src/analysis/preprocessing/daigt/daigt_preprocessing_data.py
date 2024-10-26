from dataclasses import dataclass, field
from typing import Dict, List

from src.analysis.preprocessing.common.preprocessing_data import PreprocessingData


@dataclass
class DaigtPreprocessingResults:
    
    full: Dict[str, PreprocessingData] = field(init=False, default_factory=dict)              # [collection]
    chunks: Dict[str, List[PreprocessingData]] = field(init=False, default_factory=dict)      # [collection][chunk_id]