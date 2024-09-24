from typing import List
from src.analysis.analysis_data import AnalysisData
from src.models.author import Author
from src.models.collections.collection import Collection


class MetadataAnalysis:

    @staticmethod
    def get_percentage_of_removed_text(authors: List[Author]) -> float:
        raw_text_length = 0
        cleaned_text_length = 0
        for author in authors:
            raw_text_length += MetadataAnalysis._get_text_length(author.raw_collections)
            cleaned_text_length += MetadataAnalysis._get_text_length(author.cleaned_collections)
        return 100 * (raw_text_length - cleaned_text_length) / raw_text_length

    @staticmethod
    def get_cross_top_function_words_names(analysis_data: AnalysisData, top_n_function_words: int) -> List[str]:
        """Get the top n function words from all the collections"""
        cross_top_function_words_names = []
        for collection_name in analysis_data.collection_names:
            for author_name in analysis_data.author_names:
                metrics = analysis_data.collection_author_metrics[collection_name][author_name]
                cross_top_function_words_names.extend(list(metrics.sorted_function_words.keys())[:top_n_function_words])
        return list(set(cross_top_function_words_names))
    
    @staticmethod
    def _get_text_length( collections: List[Collection]) -> int:
        """Get the total number of words in the collections"""
        all_text = " ".join([collection.get_merged_text() for collection in collections])
        return len(all_text)
    
