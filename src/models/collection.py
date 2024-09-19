from pathlib import Path
from typing import List

import os
import pandas as pd

from src.models.book import Book
from src.models.llm_response import LLMResponse
from src.models.text import Text
import random
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
random.seed(42) # Fixed seed for reproducibility

class Collection():
    
    def __init__(self, name: str):
        self.name = name
        self.texts: List[Text] = []

    def read_selected_books(self, author_name: str, selected_books_csv_filepath: str, books_dir: str) -> "Collection":
        """Read selected books for the author from the directory"""
        books_filepaths = Collection._get_books_filepaths(author_name, selected_books_csv_filepath, books_dir)
        for filepath in books_filepaths:
            self.texts.append(Book(filepath))
    
    def read_generated_texts(self, author_name: str, model_data_dir: Path) -> "Collection":
        """Read generated texts for the author from the directories of the models"""
        generated_texts_files = Collection._get_generated_texts_filepaths(author_name, model_data_dir)
        for filepath in generated_texts_files:
            self.texts.append(LLMResponse(filepath))

    def get_merged_text(self) -> str:
        """Get the merged text of all texts in the collection"""
        return " ".join([text.text for text in self.texts])
       
    def get_text_chunks(self, extract_book_chunk_size: int) -> List[List[str]]:
        """Get the text chunks for the collection"""
        if self.name == "books":
            return self._get_shuffled_books_chunks(extract_book_chunk_size)
        else:
            return self._get_models_chunks()
    
    def _get_shuffled_books_chunks(self, extract_book_chunk_size: int) -> List[List[str]]:
        """To get objective author's text (not biased by a single book), books are chunked and then the chunks are shuffled"""
        chunks = []
        for t in self.texts:
            chunks.extend(Collection._chunk_text(t.text, extract_book_chunk_size))
        random.shuffle(chunks)
        return chunks
    
    def _get_models_chunks(self) -> List[List[str]]:
        """Get the chunks of the generated texts"""
        chunks = []
        for t in self.texts:
            chunks.extend(Collection._chunk_text(t.text, len(t.text)))
        return chunks
    
    @staticmethod
    def _chunk_text(text: str, max_chunk_size: int) -> List[List[str]]:
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            pass
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            # Check if adding the next sentence would exceed the chunk size
            if current_chunk_size <= max_chunk_size:
                current_chunk.append(sentence)
                current_chunk_size += len(sentence)
            else:
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_chunk_size = len(sentence)
        
        # Append the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    @staticmethod
    def _get_generated_texts_filepaths(author_name: str, model_data_dir: Path) -> List[str]:
        """Get all generated texts filepaths for a given author"""
        texts_dir = model_data_dir / author_name
        return [str(texts_dir / f) 
                for f 
                in os.listdir(texts_dir)]

    @staticmethod
    def _get_books_filepaths(author_name: str, selected_books_csv_filepath: str, books_dir: str) -> List[Path]:
        """Get selected books filepaths for a given author"""
        df = pd.read_csv(selected_books_csv_filepath)
        book_titles = df[df["author"] == author_name]["book"].tolist()
        return [Path(books_dir / (author_name + "___" + title + ".txt"))
                           for title in book_titles]
    
    def __repr__(self) -> str:
        return f"Collection({self.name})"