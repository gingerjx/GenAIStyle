from pathlib import Path
from typing import List

import pandas as pd
from src.models.book import Book
from src.models.collections.collection import Collection
import random

random.seed(42)

class BooksCollection(Collection):
    
    def __init__(self, name: str, selected_books_csv_filepath: Path, books_dir: Path):
        super().__init__(name)
        self.selected_books_csv_filepath = selected_books_csv_filepath
        self.books_dir = books_dir

    def read(self, author_name: str) -> None:
        """Read selected books for the author from the directory"""
        books_filepaths = BooksCollection._get_books_filepaths(author_name, self.selected_books_csv_filepath, self.books_dir)
        for filepath in books_filepaths:
            self.texts.append(Book(filepath))

    def get_text_chunks(self, extract_book_chunk_size: int = None) -> List[List[str]]:
        """To get objective author's text (not biased by a single book), books are chunked and then the chunks are shuffled"""
        if extract_book_chunk_size is None:
            raise ValueError("extract_book_chunk_size must be provided for BooksCollection")
        chunks = []
        for t in self.texts:
            chunks.extend(Collection._chunk_text(t.text, extract_book_chunk_size))
        random.shuffle(chunks)
        return chunks
    
    @staticmethod
    def _get_books_filepaths(author_name: str, selected_books_csv_filepath: str, books_dir: str) -> List[Path]:
        """Get selected books filepaths for a given author"""
        df = pd.read_csv(selected_books_csv_filepath)
        book_titles = df[df["author"] == author_name]["book"].tolist()
        return [Path(books_dir / (author_name + "___" + title + ".txt"))
                           for title in book_titles]
    
    def __repr__(self) -> str:
        return f"BooksCollection({self.name})"
        