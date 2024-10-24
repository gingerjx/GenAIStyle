from pathlib import Path
from typing import Dict, List
from src.models.collections.books_collection import BooksCollection
from src.datasets.common.collections.collection import Collection
from src.models.collections.models_collection import ModelsCollection
from src.settings import Settings

class Author():

    def __init__(self, settings: Settings, name: str):
        self.name = name
        self.paths = settings.paths
        self.configuration = settings.configuration

        self.raw_collections: List[Collection] = []
        self.cleaned_collections: List[Collection] = []

    def read_selected_books_collection(self) -> None:
        """Read books for the author from the directory"""
        raw_books = BooksCollection(
            name="books",
            selected_books_csv_filepath=self.paths.ws_selected_books_csv_filepath,
            books_dir=self.paths.ws_raw_books_dir,
            seed=self.configuration.seed
        )
        raw_books.read(self.name)
        self.raw_collections.append(raw_books)

    def read_generated_texts(self) -> None:
        """Read generated texts for the author from the directories of the models"""
        for model, model_data_dir in self.paths.ws_raw_models_dirs.items():
            raw_collection = ModelsCollection(
                name=f"{model}",
                model_data_dir=model_data_dir
            )
            raw_collection.read(self.name)
            self.raw_collections.append(raw_collection)

    def __repr__(self) -> str:
        return f"Author({self.name})"