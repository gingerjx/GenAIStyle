from typing import Tuple
import pandas as pd


class BaseClassification:

    @staticmethod
    def _transform_data_for_collection_classification(pca_analysis_results_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = pca_analysis_results_data.copy()
        df = df.drop(columns=['source_name', 'author_name', 'chunk_id'])
        X = df.drop(columns=['collection_name'])
        y = df['collection_name']
        return X, y