import pandas as pd
import shap
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.common.metrics_data import MetricsAnalysisResults
from src.classification.common.pca_classification import BaseClassification
from src.classification.common.pca_classification_data import ClassificationData
import xgboost as xgb

from src.settings import Settings

class AllFeaturesXGBoostClassification(BaseClassification):

    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        self.configuration = settings.configuration
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

    def _fit_and_predict_all_chunks_binary_classification(self, 
        metrics_analysis_results: MetricsAnalysisResults,
        transform_function,
    ) -> ClassificationData:
        chunks_df = self._get_chunks_dataframe(metrics_analysis_results)

        X, y = transform_function(chunks_df)
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

        self._explain_prediction(
            model=xgb_classifier, 
            X=X_test
        )
        
        return ClassificationData(
                report=report,
                accuracy=accuracy,
                model=xgb_classifier,
                X=X_test,
                y=y_test
            )