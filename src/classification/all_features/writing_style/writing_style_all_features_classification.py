from typing import Tuple
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.writing_style.writing_style_metrics_data import WritingStyleMetricsAnalysisResults
from src.classification.all_features.writing_style.writing_style_all_features_classification_data import WritingStyleAllFeaturesClassificationResults
from src.classification.common.pca_classification import BaseClassification
from src.classification.common.pca_classification_data import ClassificationData
from src.settings import Settings
import xgboost as xgb

class WritingStyleAllFeaturesXGBoostClassification(BaseClassification):

    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        self.configuration = settings.configuration
        self.feature_extractor = feature_extractor

    def fit_and_predict(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> WritingStyleAllFeaturesClassificationResults:
        return WritingStyleAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._fit_and_predict_all_chunks_binary_classification(metrics_analysis_results)
        )

    def _fit_and_predict_all_chunks_binary_classification(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> ClassificationData:
        all_chunks = metrics_analysis_results.get_all_chunks_metrics()
        chunks_df = self.feature_extractor.get_features(all_chunks)
        chunks_df.columns = [
            str(col).replace('[', 'left_square_bracket')
                .replace(']', 'right_square_bracket')
                .replace('<', 'less_than')
                .replace('>', 'more_than') 
            for col in chunks_df.columns
        ]

        X, y = WritingStyleAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification(chunks_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.configuration.test_size, 
            random_state=self.configuration.seed
        )

        xgb_classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.configuration.seed
        )
        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return ClassificationData(
                report=report,
                accuracy=accuracy,
                model=xgb_classifier,
                X=X,
                y=y
            )

    @staticmethod
    def _transform_data_for_binary_collection_classification(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = WritingStyleAllFeaturesXGBoostClassification._transform_data_for_collection_classification(pca_analysis_results_data)
        y = y.apply(
            lambda x: 0 if x == 'books' else 1
        )
        return X, y