from pathlib import Path
from typing import List

import os
import pandas as pd

from src.models.book import Book
from src.models.llm_response import LLMResponse
from src.models.text import Text
from src.settings import Settings

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