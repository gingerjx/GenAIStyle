from dataclasses import dataclass
from typing import List

from src.analysis.preprocessing.common.preprocessing import Preprocessing
from src.analysis.preprocessing.wiriting_style.writing_style_preprocessing_data import WritingStylePreprocessingResults
from src.datasets.writing_style.writing_style_dataset import WritingStyleDataset
from src.settings import Settings

class WritingStylePreprocessing(Preprocessing):

    @dataclass
    class _SplitChunk:

        source_name: str
        splits: List[str]
        sentences: List[str]

    def __init__(self, settings: Settings) -> None:
        self.extract_book_chunk_size = settings.configuration.ws_extract_book_chunk_size
        self.analysis_number_of_words = settings.configuration.ws_extract_book_chunk_size
        self.analysis_chunk_number_of_words = settings.configuration.ws_analysis_chunk_number_of_words

    def preprocess(self, dataset: WritingStyleDataset) -> WritingStylePreprocessingResults:
        """Preprocess the data"""
        preprocessing_results = WritingStylePreprocessingResults(
            author_names=[author.name for author in dataset.authors],
            collection_names=[collection.name for collection in dataset.authors[0].cleaned_collections]
        )

        for author in dataset.authors:
            for collection in author.cleaned_collections:
                preprocessed_chunks, preprocessed_full = super().preprocess(collection)
                preprocessing_results.chunks[author.name][collection.name] = preprocessed_chunks
                preprocessing_results.full[author.name][collection.name] = preprocessed_full

        return preprocessing_results