from src.models.text import Text
import os

class Book(Text):
    AUTHOR_NAME_TITLE_SEPARATOR: str = "___"

    def __init__(self, text: str, filepath: str, title: str):
        self.title = Book._extract_title(filepath)
        self.text = text
        self.filepath = filepath
    
    def get_text(self) -> str:
        """Get the text of the book"""
        return self.text
    
    def set_text(self, text: str):
        """Set the text of the book"""
        self.text = text

    def copy(self):
        """Return a copy of the book"""
        return Book(
            text=None,
            title=self.title,
            filepath=self.filepath
        )
    
    @staticmethod
    def from_file(filepath: str) -> 'Book':
        """Create a book from a file"""
        title = Book._extract_title(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            return Book(
                text = f.read(),
                filepath = filepath,
                title=title
            )
    
    @staticmethod
    def _extract_title(path: str) -> str:
        """Extract title from a file path"""
        return os.path.basename(path) \
                .split(Book.AUTHOR_NAME_TITLE_SEPARATOR)[1] \
                .split(".")[0]