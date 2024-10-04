from dataclasses import dataclass
import pandas as pd

@dataclass
class LogisticRegressionResults:
    cross_validation_accuracy: float
    final_accuracy: float
    accuracy_per_author: pd.Series
    model: object
    report: str
    X: pd.DataFrame
    y: pd.Series