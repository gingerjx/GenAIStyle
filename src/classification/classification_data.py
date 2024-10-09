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
    def print(logistic_regression_results: LogisticRegressionResults) -> str:
        LogisticRegressionResultsTransformer._print_all_chunks_results(logistic_regression_results.all_chunks_binary_classification)
        LogisticRegressionResultsTransformer._print_author_chunks_results(logistic_regression_results.authors_chunks_binary_classification)

    @staticmethod
    def _print_all_chunks_results(logistic_regression_data: LogisticClassificationData) -> str:
        print(f"Cross-validation accuracy: {logistic_regression_data.cross_validation_accuracy}\n ---")
        print(f"Cross-validation accuracy per class:\n {logistic_regression_data.accuracy_per_class}\n ---")
        print(f"Cross-validation accuracy per author:\n {logistic_regression_data.accuracy_per_author}\n ---")

    @staticmethod
    def _print_author_chunks_results(logistic_regression_data: Dict[str, LogisticClassificationData]) -> str:
        df = pd.DataFrame()
        for author_name, results in logistic_regression_data.items():
            df = df.append(
                pd.DataFrame(
                    {
                        'author_name': author_name,
                        'accuracy_per_class': results.accuracy_per_class
                    }
                ) 
            )
        print(df)