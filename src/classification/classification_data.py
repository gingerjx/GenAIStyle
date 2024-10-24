from dataclasses import dataclass
from typing import Dict
import pandas as pd
from IPython.display import display_html

@dataclass
class ClassificationData:
    accuracy: float
    report: str
    model: object
    X: pd.DataFrame
    y: pd.Series

@dataclass
class ClassificationResults:
    # Binary classifications

    # Results of Classification performed on all chunks, all authors and collections are included in the pca.
    all_chunks_binary_classification: ClassificationData
    # Results of Classification performed seperately for each author, all author's chunks are included in the pca.
    authors_chunks_binary_classification: Dict[str, ClassificationData] # [author]
    # Results of Classification performed seperately for each collection, all collection's chunks are included in the pca.
    collections_chunks_binary_classification: Dict[str, ClassificationData] # [collection]  
    # Results of Classification performed separately for each collection-collection-author, all chunks are included in the pca. 
    collection_collection_author_chunks_classification: Dict[str, Dict[str, Dict[str, ClassificationData]]] # [author][collection][collection]
    # collection_vs_collection_per_author_classification without duplicates (collection1 vs collection2 and collection2 vs collection1)
    collection_collection_author_chunks_classification_triangle: Dict[str, Dict[str, Dict[str, ClassificationData]]] # [author][collection][collection]
    # Results of classification of author pairs withing the same collection
    collection_author_author_classification: Dict[str, Dict[str, Dict[str, ClassificationData]]] # [collection][author1][author2]
    # Author classifications
    all_chunks_binary_classification: ClassificationData


class ClassificationResultsTransformer:

    @staticmethod
    def print_all_chunks_results(logistic_regression_data: ClassificationData) -> str:
        print(f"Accuracy: {logistic_regression_data.accuracy}")

    @staticmethod
    def print_authors_chunks_results(logistic_regression_data: Dict[str, ClassificationData]) -> str:
        for author_name, results in logistic_regression_data.items():
            print(f"Accuracy for {author_name}:  {results.accuracy}")
        print(f"\n --- \nOverall accuracy: {sum([results.accuracy for results in logistic_regression_data.values()]) / len(logistic_regression_data)}")

    @staticmethod
    def print_collection_chunks_results(logistic_regression_data: Dict[str, ClassificationData]) -> str:
        for collection_name, results in logistic_regression_data.items():
            print(f"Accuracy for {collection_name}:  {results.accuracy}")
        print(f"\n --- \nOverall accuracy: {sum([results.accuracy for results in logistic_regression_data.values()]) / len(logistic_regression_data)}")


    @staticmethod
    def print_collection_collection_author_chunks_classification_results(logistic_regression_data: Dict[str, Dict[str, Dict[str, ClassificationData]]]) -> str:
        df = pd.DataFrame()
        for author_name, collections in logistic_regression_data.items():
            for collection_name_outer, collection in collections.items():
                for collection_name_inner, results in collection.items():
                    series = pd.Series([author_name, collection_name_outer, collection_name_inner, results.accuracy])
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
    
    @staticmethod
    def print_author_pairwise_classification_chunks_classification_results(classification_data: Dict[str, Dict[str, ClassificationData]]) -> pd.DataFrame:
        def map_func(data: ClassificationData):
            if data is None:
                return 1
            return data.accuracy
        
        for collection_name, authors_classification in classification_data.items():
            display_html(f"<h1>{collection_name}</h1>", raw=True)
            accuracy_df = authors_classification.map(map_func)
            accuracy_html = accuracy_df.style.background_gradient(cmap='RdYlGn').to_html()
            display_html(accuracy_html,raw=True)
        