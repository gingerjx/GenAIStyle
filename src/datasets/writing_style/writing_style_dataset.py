from src.file_utils import FileUtils
from src.models.author import Author
from src.settings import Settings


class WritingStyleDataset:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.configuration = settings.configuration
        self.paths = settings.paths

        self.author_names = FileUtils.read_authors(self.paths.ws_selected_authors_filepath)
        self.authors = []

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
        pass