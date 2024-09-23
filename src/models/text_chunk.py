from dataclasses import dataclass
from typing import List
from src.models.book import Book

@dataclass
class TextChunk:
    
    sentences: List[str]
    source_name: str = None
    