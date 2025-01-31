from dataclasses import dataclass
from typing import List

from sklearn.decomposition import PCA

from src.analysis.pca.common.pca_data import PCAAnalysisData


@dataclass
class DaigtPCAResults:
    collection_names: List[str]
    
    all_chunks: PCAAnalysisData