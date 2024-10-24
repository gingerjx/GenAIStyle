from typing import Dict, Tuple, Type
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.analysis.pca.data import PCAAnalysisResults
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.classification.classification_data import ClassificationData, ClassificationResults
from src.settings import Settings
    
class BaseClassification:

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration
        self.model_class: Type = None
        self.model_kwargs: Dict = None

    def classify(self, pca_analysis_results: PCAAnalysisResults) -> ClassificationResults:
        all_chunks_binary_classification = self._fit_and_binary_predict_on_pca(
            pca_analysis_results_data=pca_analysis_results.all_chunks.results,
            transformation_function=BaseClassification._transform_data_for_binary_collection_classification
        )
        authors_chunks_binary_classification = self._get_authors_chunks_binary_classification(pca_analysis_results)
        collections_chunks_binary_classification = self._get_collections_chunks_binary_classification(pca_analysis_results)
        collection_collection_author_chunks_classification, collection_collection_author_chunks_classification_triangle = self._get_collection_collection_author_chunks_classification(pca_analysis_results)
        collection_author_author_classification = self._get_collection_author_author_classification(pca_analysis_results)

        return ClassificationResults(
            all_chunks_binary_classification=all_chunks_binary_classification,
            authors_chunks_binary_classification=authors_chunks_binary_classification,
            collections_chunks_binary_classification=collections_chunks_binary_classification,
            collection_collection_author_chunks_classification=collection_collection_author_chunks_classification,
            collection_collection_author_chunks_classification_triangle=collection_collection_author_chunks_classification_triangle,
            collection_author_author_classification=collection_author_author_classification,
        )
    
    def _get_authors_chunks_binary_classification(self, pca_analysis_results: PCAAnalysisResults) -> Dict:
        return {
            author_name: self._fit_and_binary_predict_on_pca(
                pca_analysis_results_data=pca_analysis_results.get_authors_chunks_results(author=author_name),
                transformation_function=BaseClassification._transform_data_for_binary_collection_classification
            ) 
            for author_name in pca_analysis_results.author_names
        }
    
    def _get_collections_chunks_binary_classification(self, pca_analysis_results: PCAAnalysisResults) -> Dict:
        return {
            collection_name: self._fit_and_binary_predict_on_pca(
                pca_analysis_results_data=pca_analysis_results.get_collections_chunks_results(collection_name),
                transformation_function=BaseClassification._transform_data_for_authors_classification
            )
            for collection_name in pca_analysis_results.collection_names
        }
    
    def _get_collection_collection_author_chunks_classification(self, pca_analysis_results: PCAAnalysisResults) -> Tuple:
        result = {}
        result_trinagle = {}

        for author_name in pca_analysis_results.author_names:
            result[author_name] = {}
            result_trinagle[author_name] = {}

            for collection_name_outer in pca_analysis_results.collection_names:
                result[author_name][collection_name_outer] = {}
                result_trinagle[author_name][collection_name_outer] = {}

                for collection_name_inner in pca_analysis_results.collection_names:
                    if collection_name_outer == collection_name_inner:
                        result[author_name][collection_name_outer][collection_name_inner] = None
                        continue
                    if BaseClassification._already_classified(result[author_name], collection_name_outer, collection_name_inner):
                        result[author_name][collection_name_outer][collection_name_inner] = result[author_name][collection_name_inner][collection_name_outer]
                        continue
                    
                    output = self._fit_and_binary_predict_on_pca(
                        pca_analysis_results_data=pca_analysis_results.get_author_collection_collection_chunks_results(author_name, collection_name_outer, collection_name_inner),
                        transformation_function=BaseClassification._transform_data_for_collection_classification
                    )
                    result_trinagle[author_name][collection_name_outer][collection_name_inner] = result[author_name][collection_name_outer][collection_name_inner] = output

        return result, result_trinagle

    def _get_collection_author_author_classification(self, pca_analysis_results: PCAAnalysisResults) -> Dict:
        tables = {}

        for collection_name in pca_analysis_results.collection_names:
            tables[collection_name] = pd.DataFrame(index=pca_analysis_results.author_names, columns=pca_analysis_results.author_names)

            for outer_author_name in pca_analysis_results.author_names:
                for inner_author_name in pca_analysis_results.author_names:
                    if outer_author_name == inner_author_name:
                        tables[collection_name].at[outer_author_name, inner_author_name] = None
                        continue
                    if not pd.isnull(tables[collection_name].at[inner_author_name, outer_author_name]):
                        tables[collection_name].at[outer_author_name, inner_author_name] = tables[collection_name].at[inner_author_name, outer_author_name]
                        continue

                    outer_results = pca_analysis_results.get_collection_author_chunks_results(collection_name, outer_author_name)
                    inner_results = pca_analysis_results.get_collection_author_chunks_results(collection_name, inner_author_name)
                    tables[collection_name].at[outer_author_name, inner_author_name] = self._fit_and_binary_predict_on_pca(
                        pca_analysis_results_data=pd.concat([outer_results, inner_results]),
                        transformation_function=BaseClassification._transform_data_for_authors_classification
                    )

        return tables
    
    def _fit_and_binary_predict_on_pca(self, pca_analysis_results_data: pd.DataFrame, transformation_function) -> ClassificationData:
        X, y = transformation_function(pca_analysis_results_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.configuration.test_size, random_state=self.configuration.seed)
        
        model = self.model_class(**self.model_kwargs).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        return ClassificationData(
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
    def _already_classified(result: dict, collection_name_outer: str, collection_name_inner: str) -> bool:
        return collection_name_inner in result and collection_name_outer in result[collection_name_inner]
    
class LogisticRegressionClassification(BaseClassification):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model_class = LogisticRegression
        self.model_kwargs = {}

class SVMClassification(BaseClassification):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model_class = SVC
        self.model_kwargs = {}

class DecisionTreeClassification(BaseClassification):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model_class = DecisionTreeClassifier
        self.model_kwargs = {}