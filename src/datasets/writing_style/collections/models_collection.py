import os
from pathlib import Path
from typing import List
from src.datasets.common.collections.collection import Collection
from src.datasets.writing_style.texts.llm_response import LLMResponse
from src.datasets.common.texts.text_chunk import TextChunk


class ModelsCollection(Collection):

    def __init__(self, name: str, model_data_dir: Path):
        super().__init__(name)
        self.model_data_dir = model_data_dir

    def read(self, author_name: str) -> None:
        """Read generated texts for the author from the directories of the models"""
        generated_texts_files = ModelsCollection._get_generated_texts_filepaths(
            author_name=author_name, 
            model_data_dir=self.model_data_dir
        )
        for filepath in generated_texts_files:
            self.texts.append(LLMResponse.from_file(filepath))

    def get_text_chunks(self, chunk_size: int = None) -> List[TextChunk]:
        """Get the chunks of the generated texts"""
        chunks = []
        for response in self.texts:
            chunks.extend(ModelsCollection._chunk_text(response, len(response.text)))
        return chunks

    @staticmethod
    def _chunk_text(response: LLMResponse, chunk_size: int) -> List[TextChunk]:
        chunks_sentences = Collection._chunk_text(response.text, chunk_size)
        return [TextChunk(sentences=sentences, source_name=response.query[:50]) for sentences in chunks_sentences]
    
    @staticmethod
    def _get_generated_texts_filepaths(author_name: str, model_data_dir: Path) -> List[str]:
        """Get all generated texts filepaths for a given author"""
        texts_dir = model_data_dir / author_name
        return [str(texts_dir / f) 
                for f 
                in os.listdir(texts_dir)]
     
    def __repr__(self) -> str:
        return f"Collection({self.name})"