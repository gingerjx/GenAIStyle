from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PreprocessingData:

    source_name: str = "*"

    text: str = ""
    split: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    complex_words: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    num_of_syllabes: int = 0
    num_of_complex_words: int = 0
    num_of_sentences: int = 0
    num_of_words: int = 0

    def append_data(self, data: 'PreprocessingData') -> None:
        self.text += data.text
        self.split += data.split
        self.words += data.words
        self.complex_words += data.complex_words
        self.sentences += data.sentences
        
        self.num_of_syllabes += data.num_of_syllabes
        self.num_of_complex_words += data.num_of_complex_words
        self.num_of_sentences += data.num_of_sentences
        self.num_of_words += data.num_of_words

    def __post_init__(self) -> None:
        self.num_of_complex_words = len(self.complex_words)
        self.num_of_sentences = len(self.sentences)
        self.num_of_words = len(self.words)

@dataclass
class PreprocessingResults:
    
    author_names: List[str]
    collection_names: List[str]
    full: Dict[str, Dict[str, PreprocessingData]] = field(init=False, default_factory=dict)              # [author][model]
    chunks: Dict[str, Dict[str, List[PreprocessingData]]] = field(init=False, default_factory=dict)      # [author][model][chunk_id]

    def __post_init__(self) -> None:
        for author_name in self.author_names:   
            self.chunks[author_name] = {}
            self.full[author_name] = {}
            for collection_name in self.collection_names:
                self.full[author_name][collection_name] = PreprocessingData()
                self.chunks[author_name].update({
                    collection_name: []
                })