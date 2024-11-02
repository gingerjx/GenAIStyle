from typing import Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.analysis.pca.daigt.daigt_pca_data import DaigtPCAResults
from src.classification.common.pca_classification import BaseClassification
from src.classification.common.pca_classification_data import ClassificationData
from src.classification.writing_style.writing_style_classification_data import WritingStyleClassificationResults
from src.settings import Settings


class DaigtBaseClassification(BaseClassification):

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration

    def classify(self, pca_results: DaigtPCAResults) -> ClassificationData:
        return self._predict(
            pca_results_data=pca_results.all_chunks.results,
            transformation_function=DaigtBaseClassification._transform_data_for_binary_collection_classification
        )

    def _predict(self, pca_results_data: pd.DataFrame, transformation_function):
        X, y = transformation_function(pca_results_data)
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        
        return ClassificationData(
            report=report,
            accuracy=accuracy,
            model=self.model,
            X=X,
            y=y
        )
        
    @staticmethod
    def _transform_data_for_binary_collection_classification(pca_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = DaigtBaseClassification._transform_data_for_collection_classification(pca_results_data)
        y = y.apply(
            lambda x: 'human' if x == 'human' else 'llm'
        )
        return X, y
    
class DaigtLogisticRegressionClassification(DaigtBaseClassification):

    def __init__(self, settings: Settings, writing_style_results: WritingStyleClassificationResults):
        super().__init__(settings)
        self.model = writing_style_results.all_chunks_binary_classification.model

class DaigtSVMClassification(DaigtBaseClassification):

    def __init__(self, settings: Settings, writing_style_results: WritingStyleClassificationResults):
        super().__init__(settings)
        self.model = writing_style_results.all_chunks_binary_classification.model

class DaigtDecisionTreeClassification(DaigtBaseClassification):

    def __init__(self, settings: Settings, writing_style_results: WritingStyleClassificationResults):
        super().__init__(settings)
        self.model = writing_style_results.all_chunks_binary_classification.model