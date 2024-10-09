from typing import Tuple
from sklearn.linear_model import LogisticRegression
from src.analysis.pca.data import PCAAnalysisData, PCAAnalysisResults
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import numpy as np
from src.classification.classification_data import LogisticClassificationData, LogisticRegressionResults
from src.settings import Settings

class LogisticRegressionClassification:

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration

    def classify(self, pca_analysis_results: PCAAnalysisResults) -> LogisticRegressionResults:
        all_chunks_binary_classification = self._fit_and_binary_predict_on_pca(pca_analysis_results.all_chunks.results)
        authors_chunks_binary_classification = {
            author_name: self._fit_and_binary_predict_on_pca(pca_analysis_data.results) 
            for author_name, pca_analysis_data 
            in pca_analysis_results.collections_per_author_chunks.items()
        }
        collection_vs_collection_per_author_classification, collection_vs_collection_per_author_classification_triangle = self._get_collection_vs_collection_per_author_classification(pca_analysis_results)

        return LogisticRegressionResults(
            all_chunks_binary_classification=all_chunks_binary_classification,
            authors_chunks_binary_classification=authors_chunks_binary_classification,
            collection_vs_collection_per_author_classification=collection_vs_collection_per_author_classification,
            collection_vs_collection_per_author_classification_triangle=collection_vs_collection_per_author_classification_triangle
        )

    def _get_collection_vs_collection_per_author_classification(self, pca_analysis_results: PCAAnalysisResults) -> LogisticRegressionResults:
        result = {}
        result_trinagle = {}

        for author_name, collections in pca_analysis_results.collection_vs_collection_per_author_chunks.items():
            result[author_name] = {}
            result_trinagle[author_name] = {}

            for collection_name_outer, collection in collections.items():
                result[author_name][collection_name_outer] = {}
                result_trinagle[author_name][collection_name_outer] = {}

                for collection_name_inner, pca_analysis_data in collection.items():
                    if LogisticRegressionClassification._already_classified(result[author_name], collection_name_outer, collection_name_inner):
                        result[author_name][collection_name_outer][collection_name_inner] = result[author_name][collection_name_inner][collection_name_outer]
                        continue
                    output = self._fit_and_binary_predict_on_pca(
                        pca_analysis_results_data=pca_analysis_data.results,
                        use_original_labels=True
                    )
                    result_trinagle[author_name][collection_name_outer][collection_name_inner] = result[author_name][collection_name_outer][collection_name_inner] = output

        return result, result_trinagle

    def _fit_and_binary_predict_on_pca(self, pca_analysis_results_data: pd.DataFrame, use_original_labels: bool = False) -> LogisticClassificationData:
        X, y = LogisticRegressionClassification._transform_data_for_two_classes(
            pca_analysis_results_data=pca_analysis_results_data,
            use_original_labels=use_original_labels
        )
        accuracy_per_author, accuracy_per_class = self._get_cross_validation(
            X=X, 
            y=y,
            author_names=LogisticRegressionClassification._get_author_names_column(pca_analysis_results_data)
        )
        
        return LogisticClassificationData(
            cross_validation_accuracy=np.average(list(accuracy_per_class.to_dict().values())),
            accuracy_per_author=accuracy_per_author,
            accuracy_per_class=accuracy_per_class,
            model=LogisticRegression().fit(X, y),
            X=X,
            y=y
        )
    
    def _get_cross_validation(self, X: pd.DataFrame, y: pd.Series, author_names: pd.Series) -> float:
        skf = StratifiedKFold(n_splits=self.configuration.number_of_cv_folds, shuffle=True, random_state=self.configuration.seed)
        y_pred = cross_val_predict(LogisticRegression(), X, y, cv=skf)
        df = pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'author_name': author_names})
        
        accuracy_per_author = df.groupby('author_name').apply(
            lambda x: accuracy_score(x['y_true'], x['y_pred'])
        )
        accuracy_per_class = df.groupby('y_true').apply(
            lambda x: accuracy_score(x['y_true'], x['y_pred'])
        )

        return accuracy_per_author, accuracy_per_class
    
    @staticmethod
    def _get_author_names_column(pca_analysis_data_results: pd.DataFrame) -> pd.Series:
        return pca_analysis_data_results['author_name']

    @staticmethod
    def _transform_data_for_two_classes(pca_analysis_results_data: pd.DataFrame, use_original_labels: bool) -> Tuple[pd.DataFrame, pd.Series]:
        df = pca_analysis_results_data.copy()
        if not use_original_labels:
            df['collection_name'] = df['collection_name'].apply(
                lambda x: 'human' if x == 'books' else 'llm'
            )
        df = df.drop(columns=['source_name', 'author_name'])
        X = df.drop(columns=['collection_name'])
        y = df['collection_name']
        return X, y

    @staticmethod
    def _already_classified(result, collection_name_outer: str, collection_name_inner: str) -> bool:
        return collection_name_inner in result and collection_name_outer in result[collection_name_inner]