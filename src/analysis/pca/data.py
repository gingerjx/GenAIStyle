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

@dataclass
class PCAAnalysisResults:
    author_names: List[str]
    collection_names: List[str]
    # Results of PCA performed separately for each collection-collection-author, all chunks are included in the pca. 
    collection_vs_collection_per_author_chunks: Dict[str, Dict[str, Dict[str, PCAAnalysisData]]] # [author][collection][collection]
    # Results of PCA performed seperately for each author, all chunks are included in the pca.
    collections_per_author_chunks: Dict[str, PCAAnalysisData] # [author]