from typing import Tuple
import pandas as pd
import shap
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.daigt.daigt_metrics_data import DaigtMetricsAnalysisResults
from src.classification.all_features.daigt.daigt_all_features_classification_data import DaigtAllFeaturesClassificationResults
from src.classification.all_features.writing_style.writing_style_all_features_classification_data import WritingStyleAllFeaturesClassificationResults
from src.classification.common.pca_classification import BaseClassification
from src.classification.common.pca_classification_data import ClassificationData
from src.settings import Settings
import xgboost as xgb
from time import strftime, localtime

class DaigtAllFeaturesXGBoostClassification(BaseClassification):
            
    def __init__(self, 
                 settings: Settings, 
                 feature_extractor: FeatureExtractor,
                 ws_xgboost_results: WritingStyleAllFeaturesClassificationResults) -> None:
        self.configuration = settings.configuration
        self.feature_extractor = feature_extractor
        self.ws_xgboost_results = ws_xgboost_results

    def predict(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> DaigtAllFeaturesClassificationResults:
        return DaigtAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._predict_all_chunks_binary_classification(metrics_analysis_results)
        )

    def _predict_all_chunks_binary_classification(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> ClassificationData:
        all_chunks = metrics_analysis_results.get_all_chunks_metrics()
        chunks_df = self.feature_extractor.get_features(all_chunks)
        chunks_df.columns = [
            str(col).replace('[', 'left_square_bracket')
                .replace(']', 'right_square_bracket')
                .replace('<', 'less_than')
                .replace('>', 'more_than') 
            for col in chunks_df.columns
        ]

        X, y = DaigtAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification(chunks_df)

        xgb_classifier = self.ws_xgboost_results.all_chunks_binary_classification.model
        y_pred = xgb_classifier.predict(X)

        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)

        explainer = shap.Explainer(xgb_classifier)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, feature_names=X.columns)
        
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