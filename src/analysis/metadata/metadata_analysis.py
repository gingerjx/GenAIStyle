from typing import List
from src.datasets.daigt.daigt_dataset import DaigtDataset
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
    def daigt_get_percentage_of_removed_text(dataset: DaigtDataset) -> float:
        raw_text_length = MetadataAnalysis._get_text_length(dataset.raw_collections)
        cleaned_text_length = MetadataAnalysis._get_text_length(dataset.cleaned_collections)
        return 100 * (raw_text_length - cleaned_text_length) / raw_text_length
    
    @staticmethod
    def _get_text_length( collections: List[Collection]) -> int:
        """Get the total number of words in the collections"""
        all_text = " ".join([collection.get_merged_text() for collection in collections])
        return len(all_text)
    
