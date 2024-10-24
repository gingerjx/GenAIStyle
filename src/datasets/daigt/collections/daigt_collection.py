import random
from typing import List

from pandas import DataFrame
from src.datasets.daigt.texts.daigt_text import DaigtText
from src.datasets.common.collections.collection import Collection
from src.datasets.common.texts.text_chunk import TextChunk


class DaigtCollection(Collection):

    def __init__(self, name: str, seed: int):
        super().__init__(name)
        self.seed = seed
        random.seed(seed)

    def read(self, data: DataFrame) -> None:
        for index, row in data.iterrows():
            text_obj = DaigtText.from_series(row)
            self.texts.append(text_obj)
 
    def get_text_chunks(self, chunk_size: int = None) -> List[TextChunk]:
        chunks = []
        for text in self.texts:
            chunks.extend(DaigtCollection._chunk_text(text, len(text.get_text())))
        random.shuffle(chunks)
        return chunks
    
    @staticmethod
    def _chunk_text(text: DaigtText, chunk_size: int) -> List[TextChunk]:
        chunks_sentences = Collection._chunk_text(text.get_text(), chunk_size)
        return [TextChunk(sentences=sentences, source_name=text.prompt_name) for sentences in chunks_sentences]