from typing import List
from src.models.text import Text
from nltk.tokenize import sent_tokenize
from abc import ABC, abstractmethod

from src.models.text_chunk import TextChunk

class Collection(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.texts: List[Text] = []
    
    @abstractmethod
    def read(self, author_name: str) -> None:
        pass

    @abstractmethod  
    def get_text_chunks(self, chunk_size: int = None) -> List[TextChunk]:
        """Get the text chunks for the collection"""
        pass

    def get_merged_text(self) -> str:
        """Get the merged text of all texts in the collection"""
        return " ".join([text.text for text in self.texts])
    
    @staticmethod
    def _chunk_text(text: str, max_chunk_size: int) -> List[List[str]]:
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            pass
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            # Check if adding the next sentence would exceed the chunk size
            if current_chunk_size <= max_chunk_size:
                current_chunk.append(sentence)
                current_chunk_size += len(sentence)
            else:
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_chunk_size = len(sentence)
        
        # Append the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def __repr__(self) -> str:
        return f"Collection({self.name})"