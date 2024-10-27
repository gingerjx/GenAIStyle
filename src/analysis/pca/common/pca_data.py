from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

@dataclass
class PCAAnalysisData:
    pca: PCA = None
    source_name = None
    data: dict = None
    results: pd.DataFrame = None
    pc_variance: List[float] = None
    top_features: Dict[str, List[str]] = None
    scaler: StandardScaler = None