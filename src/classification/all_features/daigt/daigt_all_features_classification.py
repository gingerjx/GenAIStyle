from typing import Tuple
import pandas as pd
import shap
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.daigt.daigt_metrics_data import DaigtMetricsAnalysisResults
from src.classification.all_features.common.all_features_classification import AllFeaturesXGBoostClassification
from src.classification.all_features.daigt.daigt_all_features_classification_data import DaigtAllFeaturesClassificationResults
from src.classification.all_features.writing_style.writing_style_all_features_classification_data import WritingStyleAllFeaturesClassificationResults
from src.classification.common.pca_classification_data import ClassificationData
from src.settings import Settings

class DaigtAllFeaturesXGBoostClassification(AllFeaturesXGBoostClassification):
            
    def __init__(self, 
                 settings: Settings, 
                 feature_extractor: FeatureExtractor,
                 ws_xgboost_results: WritingStyleAllFeaturesClassificationResults) -> None:
        super().__init__(feature_extractor)
        self.configuration = settings.configuration
        self.ws_xgboost_results = ws_xgboost_results

    def predict(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> DaigtAllFeaturesClassificationResults:
        return DaigtAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._predict_all_chunks_binary_classification(metrics_analysis_results)
        )

    def _predict_all_chunks_binary_classification(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> ClassificationData:
        chunks_df = self._get_chunks_dataframe(metrics_analysis_results)

        X, y = DaigtAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification(chunks_df)

        xgb_classifier = self.ws_xgboost_results.all_chunks_binary_classification.model
        y_pred = xgb_classifier.predict(X)

        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)

        self._explain_prediction(
            model=xgb_classifier, 
            X=X
        )
        
        return ClassificationData(
                report=report,
                accuracy=accuracy,
                model=xgb_classifier,
                X=X,
                y=y
            )
    
    @staticmethod
    def _transform_data_for_binary_collection_classification(pca_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = DaigtAllFeaturesXGBoostClassification._transform_data_for_collection_classification(pca_results_data)
        y = y.apply(
            lambda x: 0 if x == 'human' else 1
        )
        return X, y