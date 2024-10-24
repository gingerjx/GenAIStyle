from typing import List
from src.analysis.metadata.metadata_analysis import MetadataAnalysis
from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.datasets.writing_style.writing_style_dataset import WritingStyleDataset
from src.models.author import Author
from src.datasets.common.collections.collection import Collection


class WritingStyleMetadataAnalysis(MetadataAnalysis):

    @staticmethod
    def get_percentage_of_removed_text(dataset: WritingStyleDataset) -> float:
        raw_text_length = 0
        cleaned_text_length = 0
        for author in dataset.authors:
            raw_text_length += MetadataAnalysis._get_text_length(author.raw_collections)
            cleaned_text_length += MetadataAnalysis._get_text_length(author.cleaned_collections)
        return 100 * (raw_text_length - cleaned_text_length) / raw_text_length
