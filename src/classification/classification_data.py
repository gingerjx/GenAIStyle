from dataclasses import dataclass
import pandas as pd

@dataclass
class LogisticRegressionResults:
    cross_validation_accuracy: float
    accuracy_per_author: pd.Series
    accuracy_per_class: pd.Series
    model: object
    X: pd.DataFrame
    y: pd.Series