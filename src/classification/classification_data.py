from dataclasses import dataclass
from typing import Dict, List
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
    # Results of Logistic Regresson performed on all chunks, all authors and collections are included in the pca.
    all_chunks_binary_classification: LogisticClassificationData
    # Results of PCA performed seperately for each author, all chunks are included in the pca.
    authors_chunks_binary_classification: Dict[str, LogisticClassificationData] # [author]

class LogisticRegressionResultsTransformer:

    @staticmethod
    def print_all_chunks_results(logistic_regression_data: LogisticClassificationData) -> str:
        print(f"Cross-validation accuracy: {logistic_regression_data.cross_validation_accuracy}\n ---")
        print(f"Cross-validation accuracy per class:\n {logistic_regression_data.accuracy_per_class}\n ---")
        print(f"Cross-validation accuracy per author:\n {logistic_regression_data.accuracy_per_author}\n")

    @staticmethod
    def print_author_chunks_results(logistic_regression_data: Dict[str, LogisticClassificationData]) -> str:
        df = pd.DataFrame()
        total_accuracy = 0
        for author_name, results in logistic_regression_data.items():
            total_accuracy += results.cross_validation_accuracy
            series = pd.Series([author_name, results.accuracy_per_class['llm'], results.accuracy_per_class['human']])
            df = pd.concat([df, series.to_frame().T])
        print(f"Average cross-validation accuracy: {total_accuracy / len(logistic_regression_data)}\n ---")
        df.reset_index()
        df.columns=['author_name', 'accuracy_llm', 'accuracy_human']
        print(f"Cross-validation accuracy for each author PCA:\n {df}")