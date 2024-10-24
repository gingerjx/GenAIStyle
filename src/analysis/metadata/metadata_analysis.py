from typing import List
from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.datasets.writing_style.author import Author
from src.datasets.common.collections.collection import Collection


class MetadataAnalysis:
    
    @staticmethod
    def _get_text_length( collections: List[Collection]) -> int:
        """Get the total number of words in the collections"""
        all_text = " ".join([collection.get_merged_text() for collection in collections])
        return len(all_text)
    
