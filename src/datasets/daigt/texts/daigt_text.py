import pandas as pd
from src.datasets.common.texts.text import Text


class DaigtText(Text):

    def __init__(self, text: str, label: str, prompt_name: str, source: str, model: str):
        self.text = text
        self.label = label
        self.prompt_name = prompt_name
        self.source = source
        self.model = model

    def get_text(self) -> str:
        return self.text
    
    def set_text(self, text: str) -> None:
        self.text = text
        
    def copy(self, text: str) -> "DaigtText":
        return DaigtText(
            text = text,
            label = self.label,
            prompt_name = self.prompt_name,
            source = self.source,
            model = self.model,
        )
    
    @staticmethod
    def from_series(series: pd.Series) -> "DaigtText":
        return DaigtText(
            text = series["text"],
            label = series["label"],
            prompt_name = series["prompt_name"],
            source = series["source"],
            model = series["model"],
        )