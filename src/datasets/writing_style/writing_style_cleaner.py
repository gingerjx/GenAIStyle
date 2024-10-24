from src.datasets.common.cleaner import Cleaner
from src.models.author import Author
from src.models.collections.books_collection import BooksCollection
from src.models.collections.models_collection import ModelsCollection
from src.settings import Settings

class WritingStyleCleaner(Cleaner):

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def clean(self, author: Author) -> None:
        """Clean the generated and books corpus"""
        for collection in author.raw_collections:
            if isinstance(collection.name, BooksCollection):
                self._clean_book_texts(collection)
            elif isinstance(collection.name, ModelsCollection):
                self._clean_generated_texts(collection)

    def _clean_book_texts(self, collection: BooksCollection) -> None:
        """Clean the generated corpus"""
        for text in collection.texts:
            cleanead_text = Cleaner._remove_italic(text.get_text())
            cleanead_text = Cleaner._remove_dividers(cleanead_text)
            cleanead_text = Cleaner._remove_illustration_annotations(cleanead_text)
            cleanead_text = Cleaner._remove_note_annotation(cleanead_text)
            text.set_text(cleanead_text)

    def _clean_generated_texts(self, collection: ModelsCollection) -> None:
        """Clean the books corpus"""
        for text in collection.texts:        
            if self._is_too_small(text.get_text()):
                text.set_text("")
                return
            if self._ends_with_repeated_substring(text.get_text()):
                text.set_text("")
                return
            cleanead_text = self._remove_emojis(text.get_text())
            cleanead_text = Cleaner._remove_ats(cleanead_text)
            cleanead_text = Cleaner._remove_html_tags(cleanead_text)
            text.set_text(cleanead_text)