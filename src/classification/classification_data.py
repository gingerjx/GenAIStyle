from dataclasses import dataclass
import pandas as pd

@dataclass
class LogisticClassificationData:
    cross_validation_accuracy: float
    accuracy_per_author: pd.Series
    accuracy_per_class: pd.Series
    model: object
    X: pd.DataFrame
    y: pd.Series

@dataclass
class LogisticRegressionResults:
    all_chunks_binary_classification: LogisticClassificationData