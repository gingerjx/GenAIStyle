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

    # Results of PCA performed on all chunks, all authors and collections are included in the pca.
    all_chunks: PCAAnalysisData
    # Results of PCA performed seperately for each author, all chunks are included in the pca.
    authors_chunks: Dict[str, PCAAnalysisData] # [author]
    # Results of PCA performed seperately for each collection, all chunks are included in the pca.
    collections_chunks: Dict[str, PCAAnalysisData] # [collection]
    # Results of PCA performed separately for each collection-collection-author, all chunks are included in the pca. 
    author_collection_collection_chunks: Dict[str, Dict[str, Dict[str, PCAAnalysisData]]] # [author][collection][collection]

    # Results of author-collection analysis, all chunks are included in the pca. Currently not used.
    author_collection_chunks: Dict[str, Dict[str, PCAAnalysisData]] = None # [author][collection]