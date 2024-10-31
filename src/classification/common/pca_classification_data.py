from dataclasses import dataclass

import pandas as pd


@dataclass
class ClassificationData:
    accuracy: float
    report: str
    model: object
    X: pd.DataFrame
    y: pd.Series