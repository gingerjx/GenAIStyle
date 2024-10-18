from typing import Dict, Tuple, Type
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from src.analysis.pca.data import PCAAnalysisResults
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.classification.classification_data import LogisticClassificationData, LogisticRegressionResults
from src.settings import Settings
    
class BaseClassification:

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration
        self.model_class: Type = None
        self.model_kwargs: Dict = None

    def classify(self, pca_analysis_results: PCAAnalysisResults) -> LogisticRegressionResults:
        all_chunks_binary_classification = self._fit_and_binary_predict_on_pca(
            pca_analysis_results_data=pca_analysis_results.all_chunks.results,
            transformation_function=BaseClassification._transform_data_for_binary_collection_classification
        )
        authors_chunks_binary_classification = self._get_authors_chunks_binary_classification(pca_analysis_results)
        collections_chunks_binary_classification = self._get_collections_chunks_binary_classification(pca_analysis_results)
        collection_collection_author_chunks_classification, collection_collection_author_chunks_classification_triangle = self._get_collection_collection_author_chunks_classification(pca_analysis_results)

        return LogisticRegressionResults(
            all_chunks_binary_classification=all_chunks_binary_classification,
            authors_chunks_binary_classification=authors_chunks_binary_classification,
            collections_chunks_binary_classification=collections_chunks_binary_classification,
            collection_collection_author_chunks_classification=collection_collection_author_chunks_classification,
            collection_collection_author_chunks_classification_triangle=collection_collection_author_chunks_classification_triangle
        )
    
    def _get_authors_chunks_binary_classification(self, pca_analysis_results: PCAAnalysisResults) -> Dict:
        return {
            author_name: self._fit_and_binary_predict_on_pca(
                pca_analysis_results_data=pca_analysis_data.results,
                transformation_function=BaseClassification._transform_data_for_binary_collection_classification
            ) 
            for author_name, pca_analysis_data 
            in pca_analysis_results.authors_chunks.items()
        }
    
    def _get_collections_chunks_binary_classification(self, pca_analysis_results: PCAAnalysisResults) -> Dict:
        return {
            collection_name: self._fit_and_binary_predict_on_pca(
                pca_analysis_results_data=pca_analysis_data.results,
                transformation_function=BaseClassification._transform_data_for_authors_classification
            )
            for collection_name, pca_analysis_data 
            in pca_analysis_results.collections_chunks.items()
        }
    
    def _get_collection_collection_author_chunks_classification(self, pca_analysis_results: PCAAnalysisResults) -> Tuple:
        result = {}
        result_trinagle = {}

        for author_name, collections in pca_analysis_results.author_collection_collection_chunks.items():
            result[author_name] = {}
            result_trinagle[author_name] = {}

            for collection_name_outer, collection in collections.items():
                result[author_name][collection_name_outer] = {}
                result_trinagle[author_name][collection_name_outer] = {}

                for collection_name_inner, pca_analysis_data in collection.items():
                    if BaseClassification._already_classified(result[author_name], collection_name_outer, collection_name_inner):
                        result[author_name][collection_name_outer][collection_name_inner] = result[author_name][collection_name_inner][collection_name_outer]
                        continue
                    output = self._fit_and_binary_predict_on_pca(
                        pca_analysis_results_data=pca_analysis_data.results,
                        transformation_function=BaseClassification._transform_data_for_collection_classification
                    )
                    result_trinagle[author_name][collection_name_outer][collection_name_inner] = result[author_name][collection_name_outer][collection_name_inner] = output

        return result, result_trinagle

    def _fit_and_binary_predict_on_pca(self, pca_analysis_results_data: pd.DataFrame, transformation_function) -> LogisticClassificationData:
        X, y = transformation_function(pca_analysis_results_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.configuration.test_size)
        
        model = self.model_class(**self.model_kwargs).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        return LogisticClassificationData(
            report=report,
            accuracy=accuracy,
            model=model,
            X=X,
            y=y
        )

    @staticmethod
    def _transform_data_for_binary_collection_classification(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y = BaseClassification._transform_data_for_collection_classification(pca_analysis_results_data)
        y = y.apply(
            lambda x: 'human' if x == 'books' else 'llm'
        )
        return X, y

    @staticmethod
    def _transform_data_for_collection_classification(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = pca_analysis_results_data.copy()
        df = df.drop(columns=['source_name', 'author_name'])
        X = df.drop(columns=['collection_name'])
        y = df['collection_name']
        return X, y
    
    @staticmethod
    def _transform_data_for_authors_classification(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = pca_analysis_results_data.copy()
        df = df.drop(columns=['source_name', 'collection_name'])
        X = df.drop(columns=['author_name'])
        y = df['author_name']
        return X, y
    
    @staticmethod
    def _already_classified(result, collection_name_outer: str, collection_name_inner: str) -> bool:
        return collection_name_inner in result and collection_name_outer in result[collection_name_inner]
    
class LogisticRegressionClassification(BaseClassification):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model_class = LogisticRegressionCV
        self.model_kwargs = {
            'cv': StratifiedKFold(n_splits=self.configuration.number_of_cv_folds, shuffle=True, random_state=self.configuration.seed),
            'max_iter': self.configuration.training_max_iter,
        }

class SVMClassification(BaseClassification):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model_class = SVC
        self.model_kwargs = {}