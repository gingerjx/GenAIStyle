from src.datasets.common.cleaner import Cleaner
from src.models.author import Author
from src.models.book import Book
from src.models.collections.books_collection import BooksCollection
from src.models.collections.models_collection import ModelsCollection
from src.models.llm_response import LLMResponse
from src.settings import Settings

class WritingStyleCleaner(Cleaner):

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def clean(self, author: Author) -> None:
        """Clean the generated and books corpus"""
        for collection in author.raw_collections:
            if isinstance(collection, BooksCollection):
                author.cleaned_collections.append(self._clean_book_texts(collection))
            elif isinstance(collection, ModelsCollection):
                author.cleaned_collections.append(self._clean_generated_texts(collection))

    def _clean_book_texts(self, collection: BooksCollection) -> BooksCollection:
        """Clean the generated corpus"""
        cleaned_collection = BooksCollection(
            name=collection.name,
            books_dir=collection.books_dir,
            selected_books_csv_filepath=collection.selected_books_csv_filepath,
            seed=collection.seed
        )

        for book in collection.texts:
            cleaned_book = Book.copy(book)
            cleanead_text = book.get_text()

            cleanead_text = Cleaner._remove_italic(cleanead_text)
            cleanead_text = Cleaner._remove_dividers(cleanead_text)
            cleanead_text = Cleaner._remove_illustration_annotations(cleanead_text)
            cleanead_text = Cleaner._remove_note_annotation(cleanead_text)
            cleaned_book.set_text(cleanead_text)
            cleaned_collection.texts.append(cleaned_book)

        return cleaned_collection

    def _clean_generated_texts(self, collection: ModelsCollection) -> ModelsCollection:
        """Clean the books corpus"""
        cleaned_collection = ModelsCollection(
            name=collection.name,
            model_data_dir=collection.model_data_dir
        )

        for i, response in enumerate(collection.texts):
            if self._is_too_small(response.get_text()):
                continue
            if self._ends_with_repeated_substring(response.get_text()):
                continue

            cleaned_response = LLMResponse.copy(response)
            cleanead_text = response.get_text()

            cleanead_text = self._remove_emojis(cleanead_text)
            cleanead_text = Cleaner._remove_ats(cleanead_text)
            cleanead_text = Cleaner._remove_html_tags(cleanead_text)
            cleaned_response.set_text(cleanead_text)
            cleaned_collection.texts.append(cleaned_response)

        return cleaned_collection