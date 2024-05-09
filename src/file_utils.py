import json
import os
from pathlib import Path
from typing import Dict, List

class FileUtils():

    @staticmethod
    def read_authors(filepath: str):
        return open(filepath, 'r', encoding='utf-8').read().split('\n')

    @staticmethod
    def extract_title(path: str) -> str:
        return os.path.basename(path) \
                .split("___")[1] \
                .split(".")[0]

    @staticmethod
    def read_book(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            title = FileUtils.extract_title(filepath)
            return {
                "title": title, 
                "content": content, 
                "filepath": filepath
            }

    @staticmethod
    def get_books_files(author_name: str, books_dir: str) -> List[Path]:
        return [books_dir / f 
                for f 
                in os.listdir(books_dir) 
                if f.startswith(author_name)]
            
    @staticmethod        
    def read_books(authors_names: List[str], books_dir: str) -> Dict[str, List[dict]]:
        author_books = {}
        for author_name in authors_names:
            author_books[author_name] = []
            books_files = FileUtils.get_books_files(author_name, books_dir)
            for book_file in books_files:
                author_books[author_name].append(FileUtils.read_book(book_file))
        return author_books

    @staticmethod
    def read_generated_text(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    @staticmethod
    def read_generated_texts(authors: List[str], root_dir: str):
        generated_text = {}
        for author in authors:
            generated_text[author] = []
            author_dir = Path(root_dir) / author
            for text_file in os.listdir(author_dir):
                generated_text[author].append(FileUtils.read_generated_text(author_dir / text_file))
        return generated_text
