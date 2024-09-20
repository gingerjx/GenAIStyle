from dataclasses import dataclass, field
from typing import List

@dataclass
class PreprocessingData:

    text: str = ""
    split: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    complex_words: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    num_of_syllabes: int = 0

    num_of_complex_words: int = field(init=False)
    num_of_sentences: int = field(init=False)
    num_of_words: int = field(init=False)

    def append_data(self, data: 'PreprocessingData') -> None:
        self.text += data.text
        self.split += data.split
        self.words += data.words
        self.complex_words += data.complex_words
        self.sentences += data.sentences
        self.num_of_syllabes += data.num_of_syllabes

    def calculate_counts(self) -> None:
        self.num_of_complex_words = len(self.complex_words)
        self.num_of_sentences = len(self.sentences)
        self.num_of_words = len(self.words)
    
    def __post_init__(self) -> None:
        self.calculate_counts()

@dataclass
class PreprocessingResults:

    full: PreprocessingData
    chunks: List[PreprocessingData]