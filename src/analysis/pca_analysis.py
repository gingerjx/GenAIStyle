from dataclasses import fields
from typing import Dict

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from src.analysis.analysis_data import AnalysisData, MetricData, PCAData
from string import punctuation

class PCAAnalysis:
    
    TOP_FEATURES: int = 10

    def get_analysis(analysis_data: AnalysisData) -> PCAData:
        pca_data = PCAAnalysis._get_pca_data(analysis_data)
        pca_results, pca_top_features, explained_variance_ratio_ = PCAAnalysis._get_pca(pca_data)
        return PCAData(data=pca_data, results=pca_results, pc_variance=explained_variance_ratio_, top_features=pca_top_features)

    @staticmethod
    def _get_pca_data(analysis_data: AnalysisData) -> pd.DataFrame:
        """Get the PCA of the analysis data"""
        processed_columns = [f.name for f in fields(MetricData)]
        processed_columns.remove("sorted_function_words")
        processed_columns.remove("punctuation_frequency")
        punctuation_columns = list(punctuation) 
        all_columns = processed_columns + punctuation_columns + analysis_data.metadata.cross_top_function_words_names
        df = pd.DataFrame([], columns=all_columns)

        for collection_name in analysis_data.collection_names:
            for metrics in analysis_data.collection_metrics[collection_name]:
                serie = [getattr(metrics, column) for column in processed_columns]
                serie.extend([metrics.punctuation_frequency[column] for column in punctuation_columns])
                serie.extend([metrics.sorted_function_words.get(column, 0)for column in analysis_data.metadata.cross_top_function_words_names])
                df.loc[len(df)] = serie
        return df
    
    @staticmethod
    def _get_pca(pca_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get the PCA of the analysis data"""
        targets = ["collection_name", "author_name"]
        features = [column for column in pca_data.columns if column not in targets]

        x = pca_data.loc[:, features].values
        x_scaled = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x_scaled)
        pc_df = pd.DataFrame(data = principal_components, columns = ["PC1", "PC2"])
        pca_df = pd.concat([pc_df, pca_data[targets]], axis = 1) 

        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
        top_features = {
            'PC1': loadings['PC1'].abs().sort_values(ascending=False).index.tolist()[:PCAAnalysis.TOP_FEATURES],
            'PC2': loadings['PC2'].abs().sort_values(ascending=False).index.tolist()[:PCAAnalysis.TOP_FEATURES]
        }

        return pca_df, top_features, pca.explained_variance_ratio_