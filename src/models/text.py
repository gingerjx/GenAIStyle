from pathlib import Path
from abc import ABC, abstractmethod

class Text(ABC):

    def __init__(self, filepath: Path):
        self.filepath = filepath

    @abstractmethod
    def get_text(self) -> str: pass

    @abstractmethod
    def set_text(self) -> None: pass
