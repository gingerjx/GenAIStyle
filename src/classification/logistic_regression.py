from typing import Tuple
from sklearn.linear_model import LogisticRegression
from src.analysis.pca.data import PCAAnalysisResults
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from src.classification.classification_data import LogisticRegressionResults
from src.settings import Settings
from sklearn.model_selection import train_test_split

class LogisticRegressionClassification:

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration

    def fit_and_predict_on_pca_for_two_classes(self, pca_analysis_results: PCAAnalysisResults) -> LogisticRegressionResults:
        X, y = LogisticRegressionClassification._transform_data_for_two_classes(pca_analysis_results.all_chunks.results)
        cross_validation_accuracy = self._get_cross_validation_score(X, y)
        X, y = LogisticRegressionClassification._transform_data_for_two_classes(pca_analysis_results.all_chunks.results, with_author_names=True)
        final_accuracy, accuracy_per_author, model, report = self._fit_and_predict(X, y)
        
        return LogisticRegressionResults(
            cross_validation_accuracy=cross_validation_accuracy,
            final_accuracy=final_accuracy,
            accuracy_per_author=accuracy_per_author,
            model=model,
            report=report,
            X=X,
            y=y
        )
    
    def _get_cross_validation_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return np.average(
            cross_val_score(
                estimator=LogisticRegression(), 
                X=X, y=y, 
                cv=self.configuration.number_of_cv_folds
            )
        )

    def _fit_and_predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, object, str]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.configuration.test_size, 
            random_state=self.configuration.seed
        )
        X_train_no_author_name, X_test_no_author_name = LogisticRegressionClassification._remove_author_name_from_data(X_train, X_test)
        model = LogisticRegression().fit(X_train_no_author_name, y_train)
        y_pred = model.predict(X_test_no_author_name)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_per_author = LogisticRegressionClassification._accuracy_score_per_author(y_test, y_pred, X_test)
        report = classification_report(y_test, y_pred)

        return accuracy, accuracy_per_author, model, report
    
    @staticmethod
    def _remove_author_name_from_data(X_train: pd.DataFrame, X_test: pd.Series) -> pd.DataFrame:
        return (
            X_train.drop(columns=['author_name']),
            X_test.drop(columns=['author_name']),
        )
                                  
    @staticmethod
    def _accuracy_score_per_author(y_true: pd.Series, y_pred: pd.Series, X_test: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred, 'author_name': X_test["author_name"]})
        accuracy_per_author = df.groupby('author_name').apply(
            lambda x: accuracy_score(x['y_true'], x['y_pred'])
        )
        return accuracy_per_author

    @staticmethod
    def _transform_data_for_two_classes(pca_analysis_results_data: pd.DataFrame, with_author_names: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = pca_analysis_results_data.copy()
        df['collection_name'] = df['collection_name'].apply(
            lambda x: 'human' if x == 'books' else 'llm'
        )
        if with_author_names:
            df = df.drop(columns=['source_name'])
        else:
            df = df.drop(columns=['source_name', 'author_name'])
        X = df.drop(columns=['collection_name'])
        y = df['collection_name']
        return X, y