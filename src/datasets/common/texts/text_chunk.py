from dataclasses import dataclass
from typing import List

@dataclass
class TextChunk:
    
    sentences: List[str]
    source_name: str = None
    