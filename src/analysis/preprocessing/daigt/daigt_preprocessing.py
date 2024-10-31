from dataclasses import dataclass
from typing import List

import nltk

from src.analysis.preprocessing.common.preprocessing import Preprocessing
from src.analysis.preprocessing.daigt.daigt_preprocessing_data import DaigtPreprocessingResults
from src.datasets.common.texts.text_chunk import TextChunk
from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.settings import Settings

class DaigtPreprocessing(Preprocessing):

    @dataclass
    class _SplitChunk:
        source_name: str
        splits: List[str]
        sentences: List[str]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.extract_book_chunk_size = None
        self.analysis_number_of_words = settings.configuration.daigt_analysis_number_of_words
        self.analysis_chunk_number_of_words = settings.configuration.daigt_analysis_chunk_number_of_words

    def preprocess(self, dataset: DaigtDataset) -> DaigtPreprocessingResults:
        """Preprocess the data"""
        preprocessing_results = DaigtPreprocessingResults(
            collection_names=[collection.name for collection in dataset.cleaned_collections]
        )

        for collection in dataset.cleaned_collections:
            preprocessed_chunks, preprocessed_full = super().preprocess(collection)
            preprocessing_results.chunks[collection.name] = preprocessed_chunks
            preprocessing_results.full[collection.name] = preprocessed_full

        return preprocessing_results
    
    def _get_split(self, text_chunks: List[TextChunk]) -> List[_SplitChunk]:
        split_chunks = []

        for chunk in text_chunks:
            current_chunk_splits = []

            for sentence in chunk.sentences:
                sentence_split = nltk.word_tokenize(sentence)
                current_chunk_splits.extend(sentence_split)

            split_chunks.append(Preprocessing._SplitChunk(
                source_name=chunk.source_name, 
                splits=current_chunk_splits, 
                sentences=chunk.sentences)
            )

        return split_chunks