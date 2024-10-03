from dataclasses import dataclass
import pandas as pd

@dataclass
class LogisticRegressionResults:
    cross_validation_accuracy: float
    final_accuracy: float
    model: object
    report: str
    X: pd.DataFrame
    y: pd.Series