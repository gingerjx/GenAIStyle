import pandas as pd
from src.models.text import Text


class DaigtText(Text):

    def __init__(self, row: pd.Series):
        super().__init__(None)
        self.text = row["text"]
        self.label = row["label"]
        self.prompt_name = row["prompt_name"]
        self.source = row["source"]
        self.model = row["model"]

    def get_text(self) -> str:
        return self.text
    
    def set_text(self, text: str) -> None:
        self.text = text
        
    def copy(self, text: str) -> "DaigtText":
        return DaigtText(pd.Series({
            "text": text,
            "label": self.label,
            "prompt_name": self.prompt_name,
            "source": self.source,
            "model": self.model
        }))