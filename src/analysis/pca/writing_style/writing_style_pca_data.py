from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.analysis.pca.common.pca_data import PCAAnalysisData


@dataclass
class WritingStylePCAAnalysisResults:
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

    def get_authors_chunks_results(self, author: str) -> PCAAnalysisData:
        """
        Get subset of all_chunks PCA analysis results for a specific author.
        """
        chunks_results = self.all_chunks.results
        return chunks_results[chunks_results['author_name'] == author]

    def get_collections_chunks_results(self, collection: str) -> PCAAnalysisData:
        """
        Get subset of all_chunks PCA analysis results for a specific collection.
        """
        chunks_results = self.all_chunks.results
        return chunks_results[chunks_results['collection_name'] == collection]
    
    def get_collection_author_chunks_results(self, collection: str, author: str) -> PCAAnalysisData:
        """
        Get subset of all_chunks PCA analysis results for a specific collection and author.
        """
        chunks_results = self.all_chunks.results
        collection_chunks = chunks_results[chunks_results['collection_name'] == collection]
        return collection_chunks[collection_chunks['author_name'] == author]
    
    def get_author_collection_collection_chunks_results(self, author: str, collection_outer: str, collection_inner: str) -> PCAAnalysisData:
        """
        Get subset of all_chunks PCA analysis results for a specific author, collection pair.
        """
        chunks_results = self.all_chunks.results
        author_chunks = chunks_results[chunks_results['author_name'] == author]
        outer_collection_chunks = author_chunks[author_chunks['collection_name'] == collection_outer]
        inner_collection_chunks = author_chunks[author_chunks['collection_name'] == collection_inner]
        return pd.concat([outer_collection_chunks, inner_collection_chunks])