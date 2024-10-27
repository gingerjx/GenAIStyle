from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class PCAAnalysisData:
    source_name = None
    data: dict = None
    results: pd.DataFrame = None
    pc_variance: List[float] = None
    top_features: Dict[str, List[str]] = None