from typing import Tuple
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.daigt.daigt_metrics_data import DaigtMetricsAnalysisResults
from src.classification.all_features.daigt.daigt_all_features_classification_data import DaigtAllFeaturesClassificationResults
from src.classification.common.pca_classification import BaseClassification
from src.classification.common.pca_classification_data import ClassificationData
from src.settings import Settings
import xgboost as xgb
from time import strftime, localtime

class DaigtAllFeaturesXGBoostClassification(BaseClassification):
            
    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        self.configuration = settings.configuration
        self.feature_extractor = feature_extractor

    def fit_and_predict(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> DaigtAllFeaturesClassificationResults:
        return DaigtAllFeaturesClassificationResults(
            all_chunks_binary_classification=self._fit_and_predict_all_chunks_binary_classification(metrics_analysis_results)
        )

    def _fit_and_predict_all_chunks_binary_classification(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> ClassificationData:
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 1")
        all_chunks = metrics_analysis_results.get_all_chunks_metrics()
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 2")
        chunks_df = self.feature_extractor.get_features(all_chunks)
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 3")
        chunks_df.columns = [
            str(col).replace('[', 'left_square_bracket')
                .replace(']', 'right_square_bracket')
                .replace('<', 'less_than')
                .replace('>', 'more_than') 
            for col in chunks_df.columns
        ]

        print(f"[{strftime('%H:%M:%S', localtime())}] Step 4")
        X, y = DaigtAllFeaturesXGBoostClassification._transform_data_for_binary_collection_classification(chunks_df)
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 5")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.configuration.test_size, 
            random_state=self.configuration.seed
        )

        print(f"[{strftime('%H:%M:%S', localtime())}] Step 6")
        xgb_classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.configuration.seed
        )
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 7")
        xgb_classifier.fit(X_train, y_train)
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 8")
        y_pred = xgb_classifier.predict(X_test)
        print(f"[{strftime('%H:%M:%S', localtime())}] Step 9")

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
    def _transform_data_for_binary_collection_classification(pca_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = DaigtAllFeaturesXGBoostClassification._transform_data_for_collection_classification(pca_results_data)
        y = y.apply(
            lambda x: 0 if x == 'human' else 1
        )
        return X, y