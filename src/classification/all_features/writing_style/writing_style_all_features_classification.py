from typing import Tuple
import pandas as pd
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.writing_style.writing_style_metrics_data import WritingStyleMetricsAnalysisResults
from src.classification.all_features.common.all_features_classification import AllFeaturesXGBoostClassification
from src.classification.all_features.writing_style.writing_style_all_features_classification_data import WritingStyleAllFeaturesClassificationResults
from src.settings import Settings

class WritingStyleAllFeaturesXGBoostClassification(AllFeaturesXGBoostClassification):

    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        super().__init__(
            settings=settings,
            feature_extractor=feature_extractor
        )

    def fit_and_predict(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> WritingStyleAllFeaturesClassificationResults:
        return WritingStyleAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._fit_and_predict_all_chunks_binary_classification(
                metrics_analysis_results=metrics_analysis_results,
                transform_function=WritingStyleAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification
            )
        )

    @staticmethod
    def _transform_data_for_binary_collection_classification(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = WritingStyleAllFeaturesXGBoostClassification._transform_data_for_collection_classification(pca_analysis_results_data)
        y = y.apply(
            lambda x: 0 if x == 'books' else 1
        )
        return X, y