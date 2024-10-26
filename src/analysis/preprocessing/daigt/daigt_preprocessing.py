from dataclasses import dataclass
from typing import List

from src.analysis.preprocessing.common.preprocessing import Preprocessing
from src.analysis.preprocessing.daigt.daigt_preprocessing_data import DaigtPreprocessingResults
from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.settings import Settings

class DaigtPreprocessing(Preprocessing):

    @dataclass
    class _SplitChunk:

        source_name: str
        splits: List[str]
        sentences: List[str]

    def __init__(self, settings: Settings) -> None:
        self.extract_book_chunk_size = None
        self.analysis_number_of_words = settings.configuration.daigt_analysis_number_of_words
        self.analysis_chunk_number_of_words = settings.configuration.daigt_analysis_chunk_number_of_words

    def preprocess(self, dataset: DaigtDataset) -> DaigtPreprocessingResults:
        """Preprocess the data"""
        preprocessing_results = DaigtPreprocessingResults()

        for collection in dataset.cleaned_collections:
            preprocessed_chunks, preprocessed_full = super().preprocess(collection)
            preprocessing_results.chunks[collection.name] = preprocessed_chunks
            preprocessing_results.full[collection.name] = preprocessed_full

        return preprocessing_results