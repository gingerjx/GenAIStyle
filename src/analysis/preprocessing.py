from typing import List

import pyphen
from src.models.author import Author
from src.settings import Settings
import nltk
import re

nltk.download('cmudict')
from nltk.corpus import cmudict

class Preprocessing:

    class Data:

        def __init__(self,
                     text: str,
                     split: List[str],
                     words: List[str],
                     syllables_count: List[int],     
                ) -> None:
            self.text = text
            self.split = split
            self.words = words
            self.syllables_count = syllables_count
             
    def __init__(self, settings: Settings, authors: List[Author]) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.authors = authors

    def preprocess(self) -> Data:
        """Preprocess the data"""
        data= {}

        for author in self.authors:
            data.update({author: {}})
            for collection in author.cleaned_collections:
                split = self._get_split(collection.get_merged_text())[:self.configuration.analysis_size]
                words = self._get_words(split)
                text = " ".join(split)
                syllables_count = self._get_syllables_count(words)
                data[author].update({
                    collection: Preprocessing.Data(
                        text=text, 
                        split=split,
                        words=words, 
                        syllables_count=syllables_count)
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
    
    def _get_syllables_count(self, words: List[str]) -> int:
        """Get the syllables from the words"""
        count = 0
        dic = pyphen.Pyphen(lang='en')
        d = cmudict.dict()
        for word in words:
            if word.lower() in d:
                count += [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
            else:
                count += len(dic.inserted(word).split("-"))
            pass
        return count