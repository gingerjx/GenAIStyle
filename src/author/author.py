from pathlib import Path
from typing import List
from src.author.collection import Collection

class Author():

    def __init__(self, name: str):
        self.name = name
        self.collections: List[Collection] = []

    def read_book_collection(self, books_dir: str) -> None:
        books = Collection("books")
        books.read_books(self.name, books_dir)
        self.collections.append(books)

    def read_generated_texts(self, models_data_dirs: List[Path]) -> None:
        for model, model_data_dir in models_data_dirs.items():
            collection = Collection(f"{model}")
            collection.read_generated_texts(self.name, model_data_dir)
            self.collections.append(collection)