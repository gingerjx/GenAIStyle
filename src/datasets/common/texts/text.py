from pathlib import Path
from abc import ABC, abstractmethod

class Text(ABC):

    @abstractmethod
    def get_text(self) -> str: pass

    @abstractmethod
    def set_text(self) -> None: pass
