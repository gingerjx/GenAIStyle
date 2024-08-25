from typing import List

import pyphen
from src.analysis.preprocessing_data import PreprocessingData
from src.models.author import Author
from src.settings import Settings
import nltk
import re

nltk.download('cmudict')
nltk.download('punkt')
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize

class Preprocessing:
             
    def __init__(self, settings: Settings, authors: List[Author]) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.authors = authors

    def preprocess(self) -> PreprocessingData:
        """Preprocess the data"""
        data= {}

        for author in self.authors:
            data.update({author: {}})
            for collection in author.cleaned_collections:
                all_text = collection.get_merged_text()[:200000] # TODO: Remove the limit afterwards
                split = self._get_split(all_text)[:self.configuration.analysis_size]
                words = self._get_words(split)
                text = " ".join(split)
                try:
                    sentences = sent_tokenize(text)
                except:
                    pass
                num_of_syllabes, complex_words = self._get_num_of_syllabes_and_complex_words(words)
                data[author].update({
                    collection: PreprocessingData(
                        text=text, 
                        split=split,
                        words=words, 
                        complex_words=complex_words,
                        sentences=sentences,
                        num_of_syllabes=num_of_syllabes)
                })

        return data

    def _get_split(self, text: str) -> List[str]:
        """Get the split from the text"""
        split = nltk.word_tokenize(text)
        return split
    
    def _get_words(self, split: List[str]) -> List[str]:
        """Get the words from the split"""
        words = [word for word in split if re.search(r'[a-zA-Z]', word)]
        return words
    
    def _get_num_of_syllabes_and_complex_words(self, words: List[str]) -> int:
        """Get the syllables from the words"""
        num_of_syllabes = 0
        complex_words = []
        dic = pyphen.Pyphen(lang='en')
        d = cmudict.dict()
        for word in words:
            if word.lower() in d:
                number = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
            else:
                number = len(dic.inserted(word).split("-"))
            if number > 2:
                complex_words.append(word)
            num_of_syllabes += number
        return num_of_syllabes, complex_words