from pathlib import Path
from typing import List

import os

from src.author.book import Book
from src.author.llm_response import LLMResponse
from src.author.text import Text

class Collection():
    
    def __init__(self, name: str):
        self.name = name
        self.texts: List[Text] = []

    def read_books(self, author_name: str, books_dir: str) -> "Collection":
        books_files = Collection._get_books_filepaths(author_name, books_dir)
        for filepath in books_files:
            self.texts.append(Book(filepath))
    
    def read_generated_texts(self, author_name: str, model_data_dir: Path) -> "Collection":
        generated_texts_files = Collection._get_generated_texts_filepaths(author_name, model_data_dir)
        for filepath in generated_texts_files:
            self.texts.append(LLMResponse(filepath))
    
    def get_merged_text(self) -> str:
        return " ".join([text.text for text in self.texts])
    
    @staticmethod
    def _get_generated_texts_filepaths(author_name: str, model_data_dir: Path) -> List[str]:
        texts_dir = model_data_dir / author_name
        return [str(texts_dir / f) 
                for f 
                in os.listdir(texts_dir)]

    @staticmethod
    def _get_books_filepaths(author_name: str, books_dir: str) -> List[Path]:
        return [books_dir / f 
                for f 
                in os.listdir(books_dir) 
                if f.startswith(author_name)]