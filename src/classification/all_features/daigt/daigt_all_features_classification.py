from typing import Tuple
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
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
        super().__init__(
            settings=settings,
            feature_extractor=feature_extractor
        )
        self.ws_xgboost_results = ws_xgboost_results

    def predict(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> DaigtAllFeaturesClassificationResults:
        return DaigtAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._predict_all_chunks_binary_classification(
                model=self.ws_xgboost_results.all_chunks_binary_classification.model,
                metrics_analysis_results=metrics_analysis_results,
                transform_function=DaigtAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification
            )
        )

    def fit_and_predict(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> DaigtAllFeaturesClassificationResults:
        return DaigtAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._fit_and_predict_all_chunks_binary_classification(
                metrics_analysis_results=metrics_analysis_results,
                transform_function=DaigtAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification
            )
        )
    
    @staticmethod
    def _transform_data_for_binary_collection_classification(pca_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = DaigtAllFeaturesXGBoostClassification._transform_data_for_collection_classification(pca_results_data)
        y = y.apply(
            lambda x: 0 if x == 'human' else 1
        )
        return X, y