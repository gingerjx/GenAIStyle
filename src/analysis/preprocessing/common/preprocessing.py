from dataclasses import dataclass
from typing import List, Tuple

import pyphen
from src.analysis.preprocessing.common.preprocessing_data import PreprocessingData
from src.analysis.preprocessing.wiriting_style.writing_style_preprocessing_data import WritingStylePreprocessingResults
from src.datasets.common.collections.collection import Collection
from src.datasets.writing_style.author import Author
from src.datasets.common.texts.text_chunk import TextChunk
from src.datasets.writing_style.writing_style_dataset import WritingStyleDataset
from src.settings import Settings
import nltk
import re
from nltk.corpus import cmudict
import time

dic = pyphen.Pyphen(lang='en')
d = cmudict.dict()

class Preprocessing:

    @dataclass
    class _SplitChunk:

        source_name: str
        splits: List[str]
        sentences: List[str]

    def __init__(self, settings: Settings) -> None:
        self.extract_book_chunk_size = None
        self.analysis_number_of_words = None
        self.analysis_chunk_number_of_words = None

    def preprocess(self, collection: Collection) -> Tuple[List[PreprocessingData], PreprocessingData]:
        """Preprocess the data"""
        preprocessed_chunks = []
        preprocessed_full = PreprocessingData()
        text_chunks = collection.get_text_chunks(self.extract_book_chunk_size)
        split_chunks = self._get_split(text_chunks)

        for split_chunk in split_chunks:
            chunk_preprocessing_data = self._get_chunk_preprocessing_data(split_chunk)
            preprocessed_chunks.append(chunk_preprocessing_data)
            preprocessed_full.append_data(chunk_preprocessing_data)

        return preprocessed_chunks, preprocessed_full
    
    def _get_chunk_preprocessing_data(self, split_chunk: _SplitChunk) -> PreprocessingData:
        text = self._get_text(split_chunk.sentences)
        words = self._get_words(split_chunk.splits)
        num_of_syllabes, complex_words = self._get_num_of_syllabes_and_complex_words(words)
        return PreprocessingData(
            source_name=split_chunk.source_name,
            text=text, 
            split=split_chunk.splits,
            words=words, 
            complex_words=complex_words,
            sentences=split_chunk.sentences,
            num_of_syllabes=num_of_syllabes
        )

    def _get_split(self, text_chunks: List[TextChunk]) -> List[_SplitChunk]:
        """Get the split from the text"""
        split_chunks = []
        current_chunk_splits = []
        current_chunk_sentences = []
        total_split_size = 0
        chunk_split_size = 0
        
        for chunk in text_chunks:
            for sentence in chunk.sentences:
                if self._analysis_limit_reached(total_split_size, chunk_split_size):
                    split_chunks.append(Preprocessing._SplitChunk(
                        source_name=chunk.source_name, 
                        splits=current_chunk_splits, 
                        sentences=current_chunk_sentences)
                    )
                    return split_chunks
                if self._chunk_analysis_limit_reached(chunk_split_size):
                    split_chunks.append(Preprocessing._SplitChunk(
                        source_name=chunk.source_name, 
                        splits=current_chunk_splits, 
                        sentences=current_chunk_sentences)
                    )
                    current_chunk_splits = []
                    current_chunk_sentences = []
                    chunk_split_size = 0

                sentence_split = nltk.word_tokenize(sentence)
                total_split_size += len(sentence_split)
                chunk_split_size += len(sentence_split)
                current_chunk_splits.extend(sentence_split)
                current_chunk_sentences.append(sentence)

        return split_chunks
    
    def _analysis_limit_reached(self, total_split_size: int, chunk_split_size: int) -> bool:
        return self.analysis_number_of_words is not None and \
            total_split_size >= self.analysis_number_of_words and \
            chunk_split_size >= self.analysis_chunk_number_of_words
    
    def _chunk_analysis_limit_reached(self, chunk_split_size: int) -> bool:
        return self.analysis_chunk_number_of_words is not None and \
            chunk_split_size >= self.analysis_chunk_number_of_words
    
    def _get_text(self, sentences: List[str]) -> str:
        return " ".join(sentences)
    
    def _get_words(self, split: List[str]) -> List[str]:
        """Get the words from the split"""
        words = [word for word in split if re.search(r'[a-zA-Z]', word)]
        return words
    
    def _get_num_of_syllabes_and_complex_words(self, words: List[str]) -> int:
        """Get the syllables from the words"""
        num_of_syllabes = 0
        complex_words = []
        for word in words:
            if word.lower() in d:
                number = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
            else:
                number = len(dic.inserted(word).split("-"))
            if number > 2:
                complex_words.append(word)
            num_of_syllabes += number
        return num_of_syllabes, complex_words