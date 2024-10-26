from typing import List

import pandas as pd
from src.datasets.writing_style.writing_style_cleaner import WritingStyleCleaner
from src.file_utils import FileUtils
from src.datasets.writing_style.author import Author
from src.settings import Settings


class WritingStyleDataset:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.configuration = settings.configuration
        self.paths = settings.paths

        self.cleaner = WritingStyleCleaner(settings)
        self.author_names = FileUtils.read_authors(self.paths.ws_selected_authors_filepath)
        self.authors: List[Author] = []

    def load(self) -> None: 
        for author_name in self.author_names:
            author = Author(
                settings=self.settings,
                name=author_name
            )
            author.read_selected_books_collection()
            author.read_generated_texts()
            self.authors.append(author)

    def clean(self) -> None:
        for author in self.authors:
            self.cleaner.clean(author)

    def info(self) -> pd.DataFrame:
        table = pd.DataFrame(index=self.author_names, columns=self._get_collection_names())
        for author in self.authors:
            for collection in author.cleaned_collections:
                collection_text = collection.get_merged_text()
                table.loc[author.name, collection.name] = len(collection_text)
        return table

    def _get_collection_names(self) -> List[str]:
        return list(self.paths.ws_raw_models_dirs.keys())