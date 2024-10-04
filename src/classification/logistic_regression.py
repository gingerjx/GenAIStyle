from typing import Tuple
from sklearn.linear_model import LogisticRegression
from src.analysis.pca.data import PCAAnalysisData, PCAAnalysisResults
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import numpy as np
from src.classification.classification_data import LogisticRegressionResults
from src.settings import Settings
from sklearn.model_selection import train_test_split

class LogisticRegressionClassification:

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration

    def fit_and_predict_on_pca_for_two_classes(self, pca_analysis_results: PCAAnalysisResults) -> LogisticRegressionResults:
        X, y = LogisticRegressionClassification._transform_data_for_two_classes(pca_analysis_results.all_chunks.results)
        accuracy_per_author, accuracy_per_class = self._get_cross_validation(
            X=X, 
            y=y,
            author_names=LogisticRegressionClassification._get_author_names_column(pca_analysis_results.all_chunks)
        )
        
        return LogisticRegressionResults(
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
    def _get_author_names_column(pca_analysis_data: PCAAnalysisData) -> pd.Series:
        return pca_analysis_data.results['author_name']

    @staticmethod
    def _transform_data_for_two_classes(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = pca_analysis_results_data.copy()
        df['collection_name'] = df['collection_name'].apply(
            lambda x: 'human' if x == 'books' else 'llm'
        )
        df = df.drop(columns=['source_name', 'author_name'])
        X = df.drop(columns=['collection_name'])
        y = df['collection_name']
        return X, y