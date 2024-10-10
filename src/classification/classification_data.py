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
    # Binary classifications

    # Results of Logistic Regresson performed on all chunks, all authors and collections are included in the pca.
    all_chunks_binary_classification: LogisticClassificationData
    # Results of Logistic Regresson performed seperately for each author, all chunks are included in the pca.
    authors_chunks_binary_classification: Dict[str, LogisticClassificationData] # [author]
    # Results of Logistic Regresson performed separately for each collection-collection-author, all chunks are included in the pca. 
    collection_vs_collection_per_author_binary_classification: Dict[str, Dict[str, Dict[str, LogisticClassificationData]]] # [author][collection][collection]
    # collection_vs_collection_per_author_classification without duplicates
    collection_vs_collection_per_author_binary_classification_triangle: Dict[str, Dict[str, Dict[str, LogisticClassificationData]]] # [author][collection][collection]

    # Author classifications
    all_chunks_binary_classification: LogisticClassificationData

class LogisticRegressionResultsTransformer:

    @staticmethod
    def print_all_chunks_results(logistic_regression_data: LogisticClassificationData) -> str:
        print(f"Cross-validation accuracy: {logistic_regression_data.cross_validation_accuracy}\n ---")
        print(f"Cross-validation accuracy per class:\n {logistic_regression_data.accuracy_per_class}\n ---")
        print(f"Cross-validation accuracy per author:\n {logistic_regression_data.accuracy_per_author}\n")

    @staticmethod
    def print_author_chunks_results(logistic_regression_data: Dict[str, LogisticClassificationData]) -> str:
        df = pd.DataFrame()
        for author_name, results in logistic_regression_data.items():
            series = pd.Series([author_name, results.accuracy_per_class['llm'], results.accuracy_per_class['human'], results.cross_validation_accuracy])
            df = pd.concat([df, series.to_frame().T])
        df.reset_index()
        df.columns=['author_name', 'accuracy_llm', 'accuracy_human', "total_accuracy"]
        print(f"Average cross-validation accuracy: {df["total_accuracy"].mean()}\n ---")
        print(f"Cross-validation accuracy for each author PCA:\n {df}")

    @staticmethod
    def print_collection_vs_collection_per_author_results(logistic_regression_data: Dict[str, Dict[str, Dict[str, LogisticClassificationData]]]) -> str:
        df = pd.DataFrame()
        for author_name, collections in logistic_regression_data.items():
            for collection_name_outer, collection in collections.items():
                for collection_name_inner, results in collection.items():

                    series = pd.Series([author_name, collection_name_outer, collection_name_inner, results.cross_validation_accuracy])
                    df = pd.concat([df, series.to_frame().T])
        df.reset_index()
        df.columns=['author_name', 'collection_1', 'collection_2', "total_accuracy"]
        print(f"Average cross-validation accuracy: {df["total_accuracy"].mean()}\n ---")

        df['collection_1'] = df['collection_1'] + " vs " + df['collection_2']
        df.rename(columns={'collection_1': 'collection vs collection'}, inplace=True)
        df = df.drop(columns=['collection_2', "author_name"])
        df = df.groupby(["collection vs collection"]).mean()
        df.columns = ['average_accuracy']
        df = df.sort_values(by='average_accuracy', ascending=False)
        return df