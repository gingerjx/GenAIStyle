from typing import List


class PreprocessingData:

    def __init__(self,
                    text: str,
                    split: List[str],
                    words: List[str],
                    complex_words: List[str],
                    sentences: List[str],
                    num_of_syllabes: List[int],     
            ) -> None:
        self.text = text
        self.split = split
        self.words = words
        self.num_of_words = len(words)
        self.complex_words = complex_words
        self.num_of_complex_words = len(complex_words)
        self.sentences = sentences
        self.num_of_sentences = len(sentences)
        self.num_of_syllabes = num_of_syllabes