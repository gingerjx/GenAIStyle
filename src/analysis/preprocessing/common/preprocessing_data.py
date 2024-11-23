from dataclasses import dataclass, field
from typing import List
import uuid

import pandas as pd

@dataclass
class PreprocessingData:

    collection_name: str

    split: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    complex_words: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    num_of_splits: int = 0
    num_of_syllabes: int = 0
    num_of_complex_words: int = 0
    num_of_sentences: int = 0
    num_of_words: int = 0

    chunk_id: str = str(uuid.uuid4())
    source_name: str = "*"
    text: str = ""

    def append_data(self, data: 'PreprocessingData') -> None:
        self.text += data.text
        self.split += data.split
        self.words += data.words
        self.complex_words += data.complex_words
        self.sentences += data.sentences
        
        self.num_of_splits += data.num_of_splits
        self.num_of_syllabes += data.num_of_syllabes
        self.num_of_complex_words += data.num_of_complex_words
        self.num_of_sentences += data.num_of_sentences
        self.num_of_words += data.num_of_words

    def info(self) -> pd.Series:
        return pd.Series({
            "num_of_syllabes": self.num_of_syllabes,
            "num_of_complex_words": self.num_of_complex_words,
            "num_of_sentences": self.num_of_sentences,
            "num_of_words": self.num_of_words,
            "num_of_splits": self.num_of_splits,
        })
    
    def __post_init__(self) -> None:
        self.num_of_complex_words = len(self.complex_words)
        self.num_of_sentences = len(self.sentences)
        self.num_of_words = len(self.words)
        self.num_of_splits = len(self.split)