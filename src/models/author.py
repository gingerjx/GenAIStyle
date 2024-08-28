from pathlib import Path
from typing import Dict, List
from src.models.collection import Collection
from src.settings import Settings

class Author():

    def __init__(self, settings: Settings, name: str):
        self.name = name
        self.paths = settings.paths

        self.raw_collections: List[Collection] = []
        self.cleaned_collections: List[Collection] = []

    def read_selected_books_collection(self) -> None:
        """Read books for the author from the directory"""
        raw_books = Collection("books")
        raw_books.read_selected_books(self.name, self.paths.selected_books_csv_filepath, self.paths.raw_books_dir)
        self.raw_collections.append(raw_books)

        cleaned_books = Collection("books")
        cleaned_books.read_selected_books(self.name, self.paths.selected_books_csv_filepath, self.paths.cleaned_books_dir)
        self.cleaned_collections.append(cleaned_books)

    def read_generated_texts(self) -> None:
        """Read generated texts for the author from the directories of the models"""
        for model, model_data_dir in self.paths.raw_models_dirs.items():
            raw_collection = Collection(f"{model}")
            raw_collection.read_generated_texts(self.name, model_data_dir)
            self.raw_collections.append(raw_collection)
        
        for model, model_data_dir in self.paths.cleaned_models_dirs.items():
            cleaned_collection = Collection(f"{model}")
            cleaned_collection.read_generated_texts(self.name, model_data_dir)
            self.cleaned_collections.append(cleaned_collection)

    def __repr__(self) -> str:
        return f"Author({self.name})"