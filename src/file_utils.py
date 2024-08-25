import json
import os
from pathlib import Path
from typing import Dict, List

import jsonpickle

from src.analysis.analysis_data import AnalysisData

class FileUtils():

    @staticmethod
    def read_authors(filepath: str):
        """Read authors from a file"""
        return open(filepath, 'r', encoding='utf-8').read().split('\n')

    @staticmethod
    def extract_title(path: str) -> str:
        """Extract title from a file path"""
        return os.path.basename(path) \
                .split("___")[1] \
                .split(".")[0]

    @staticmethod
    def read_book(filepath: str):
        """Read book from a file"""
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
        """Get all books filepaths for a given author"""
        return [books_dir / f 
                for f 
                in os.listdir(books_dir) 
                if f.startswith(author_name)]
            
    @staticmethod        
    def read_books(authors_names: List[str], books_dir: str) -> Dict[str, List[dict]]:
        """Read books for a list of authors"""
        author_books = {}
        for author_name in authors_names:
            author_books[author_name] = []
            books_files = FileUtils.get_books_files(author_name, books_dir)
            for book_file in books_files:
                author_books[author_name].append(FileUtils.read_book(book_file))
        return author_books

    @staticmethod
    def read_generated_text(filepath: str):
        """Read generated text from a file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    @staticmethod
    def read_generated_texts(authors: List[str], root_dir: str):
        """Read generated texts for a list of authors"""
        generated_text = {}
        for author in authors:
            generated_text[author] = []
            author_dir = Path(root_dir) / author
            for text_file in os.listdir(author_dir):
                generated_text[author].append(FileUtils.read_generated_text(author_dir / text_file))
        return generated_text
    
    @staticmethod
    def read_analysis_data(filepath: str) -> Dict[str, List[AnalysisData]]:
        """Read analysis data from a file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            return jsonpickle.decode(json_data)
