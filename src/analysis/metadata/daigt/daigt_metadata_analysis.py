from typing import List
from src.analysis.metadata.metadata_analysis import MetadataAnalysis
from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.datasets.writing_style.author import Author
from src.datasets.common.collections.collection import Collection


class DaigtMetadataAnalysis(MetadataAnalysis):

    @staticmethod
    def get_percentage_of_removed_text(dataset: DaigtDataset) -> float:
        raw_text_length = MetadataAnalysis._get_text_length(dataset.raw_collections)
        cleaned_text_length = MetadataAnalysis._get_text_length(dataset.cleaned_collections)
        return 100 * (raw_text_length - cleaned_text_length) / raw_text_length
    