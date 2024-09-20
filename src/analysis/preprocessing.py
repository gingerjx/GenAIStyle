from typing import Dict, List, Tuple

import pyphen
from src.analysis.preprocessing_data import PreprocessingData, PreprocessingResults
from src.models.author import Author
from src.settings import Settings
import nltk
import re
from nltk.corpus import cmudict
import time

dic = pyphen.Pyphen(lang='en')
d = cmudict.dict()

class Preprocessing:

    def __init__(self, settings: Settings, authors: List[Author]) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.authors = authors

    def preprocess(self) -> Dict[str, Dict[str, PreprocessingResults]]:
        """Preprocess the data"""
        data= {}

        for author in self.authors:
            data.update({author.name: {}})
            for collection in author.cleaned_collections:
                text_chunks = collection.get_text_chunks(self.configuration.extract_book_chunk_size)
                chunks_splits, chunks_sentences = self._get_split(text_chunks)
                chunks_preprocessing_data = []
                full_preprocessing_data = PreprocessingData()

                for chunk_splits, chunk_sentences in zip(chunks_splits, chunks_sentences):
                    chunk_preprocessing_data = self._get_chunk_preprocessing_data(chunk_splits, chunk_sentences)
                    chunks_preprocessing_data.append(chunk_preprocessing_data)
                    full_preprocessing_data.append_data(chunk_preprocessing_data)

                full_preprocessing_data.calculate_counts()
                data[author.name].update({
                    collection.name: PreprocessingResults(
                        full=full_preprocessing_data,
                        chunks=chunks_preprocessing_data)
                })

        return data
    
    def _get_chunk_preprocessing_data(self, chunk_splits: List[str], chunk_sentences) -> PreprocessingData:
        text = self._get_text(chunk_sentences)
        words = self._get_words(chunk_splits)
        num_of_syllabes, complex_words = self._get_num_of_syllabes_and_complex_words(words)
        return PreprocessingData(
            text=text, 
            split=chunk_splits,
            words=words, 
            complex_words=complex_words,
            sentences=chunk_sentences,
            num_of_syllabes=num_of_syllabes
        )

    def _get_split(self, text_chunks: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Get the split from the text"""
        chunks_splits = []
        chunks_sentences = []
        current_chunk_splits = []
        current_chunk_sentences = []
        total_split_size = 0
        chunk_split_size = 0
        
        for chunk in text_chunks:
            for sentence in chunk:
                if total_split_size >= self.configuration.analysis_number_of_words and chunk_split_size >= self.configuration.analysis_chunk_number_of_words:
                    chunks_splits.append(current_chunk_splits)
                    chunks_sentences.append(current_chunk_sentences)
                    return chunks_splits, chunks_sentences
                if chunk_split_size >= self.configuration.analysis_chunk_number_of_words:
                    chunks_splits.append(current_chunk_splits)
                    chunks_sentences.append(current_chunk_sentences)
                    current_chunk_splits = []
                    current_chunk_sentences = []
                    chunk_split_size = 0

                sentence_split = nltk.word_tokenize(sentence)
                total_split_size += len(sentence_split)
                chunk_split_size += len(sentence_split)
                current_chunk_splits.extend(sentence_split)
                current_chunk_sentences.append(sentence)

        return chunks_splits, chunks_sentences
    
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