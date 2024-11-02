import pandas as pd
import shap
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.common.metrics_data import MetricsAnalysisResults
from src.classification.common.pca_classification import BaseClassification

class AllFeaturesXGBoostClassification(BaseClassification):

    def __init__(self, feature_extractor: FeatureExtractor) -> None:
        self.feature_extractor = feature_extractor

    def _get_chunks_dataframe(self, metrics_analysis_results: MetricsAnalysisResults) -> pd.DataFrame:
        all_chunks = metrics_analysis_results.get_all_chunks_metrics()
        chunks_df = self.feature_extractor.get_features(all_chunks)

        # Ensure all feature names are strings and remove invalid characters
        chunks_df.columns = [
            str(col).replace('[', 'left_square_bracket')
                .replace(']', 'right_square_bracket')
                .replace('<', 'less_than')
                .replace('>', 'more_than') 
            for col in chunks_df.columns
        ]

        return chunks_df
    
    def _explain_prediction(self, model, X):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, feature_names=X.columns)