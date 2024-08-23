from src.models.text import Text
import os

class Book(Text):
    AUTHOR_NAME_TITLE_SEPARATOR: str = "___"

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.title = Book._extract_title(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
    
    def text(self):
        """Get the text of the book"""
        return self.text
    
    @staticmethod
    def _extract_title(path: str) -> str:
        """Extract title from a file path"""
        return os.path.basename(path) \
                .split(Book.AUTHOR_NAME_TITLE_SEPARATOR)[1] \
                .split(".")[0]