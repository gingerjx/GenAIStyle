from dataclasses import fields
from typing import Dict, List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from src.analysis.metrics.models import MetricData, MetricsAnalysisResults
from src.analysis.pca.data import PCAAnalysisData, PCAAnalysisResults
from string import punctuation

from src.settings import Settings

class PCAAnalysis:
    
    TOP_FEATURES: int = 10

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration

    def fit_transform(self, metrics_analysis_results: MetricsAnalysisResults) -> PCAAnalysisResults:
        collection_vs_collection_per_author_chunks = self._fit_transform_pca_collection_vs_collection_per_author_chunks_analysis(metrics_analysis_results)
        collections_per_author_chunks = self._fit_transform_collections_per_author_chunks(metrics_analysis_results)
        all_chunks = self._fit_transform_all_chunks(metrics_analysis_results)
        author_collection_chunks = self._fit_transform_author_collection_chunks(metrics_analysis_results)

        return PCAAnalysisResults(
            author_names=metrics_analysis_results.author_names,
            collection_names=metrics_analysis_results.collection_names,
            collection_vs_collection_per_author_chunks=collection_vs_collection_per_author_chunks,
            collections_per_author_chunks=collections_per_author_chunks,
            all_chunks=all_chunks,
            author_collection_chunks=author_collection_chunks,
        )        
    
    def transform(self, metrics_analysis_results: MetricsAnalysisResults, pca_analysis_results: PCAAnalysisResults) -> PCAAnalysisResults:
        all_chunks = self._transform_all_chunks(metrics_analysis_results, pca_analysis_results.all_chunks)

        return PCAAnalysisResults(
            author_names=metrics_analysis_results.author_names,
            collection_names=metrics_analysis_results.collection_names,
            collection_vs_collection_per_author_chunks=None,
            collections_per_author_chunks=None,
            all_chunks=all_chunks,
            author_collection_chunks=None,
        )   

    def _fit_transform_pca_collection_vs_collection_per_author_chunks_analysis(self, metrics_analysis_results: MetricsAnalysisResults) -> Dict[str, Dict[str, PCAAnalysisData]]:       
        collection_vs_collection_per_author_analysis = {} 

        for author_name in metrics_analysis_results.author_names:
            collection_vs_collection_per_author_analysis[author_name] = {}
            for collection_name_outer in metrics_analysis_results.collection_names:
                collection_vs_collection_per_author_analysis[author_name][collection_name_outer] = {}
                for collection_name_inner in metrics_analysis_results.collection_names:
                    if collection_name_outer == collection_name_inner:
                        continue
                    if PCAAnalysis._pca_already_analysed(collection_vs_collection_per_author_analysis[author_name], collection_name_outer, collection_name_inner):
                        collection_vs_collection_per_author_analysis[author_name][collection_name_outer][collection_name_inner] = collection_vs_collection_per_author_analysis[author_name][collection_name_inner][collection_name_outer]
                        continue

                    chunks_inner = metrics_analysis_results.chunks_author_collection[author_name][collection_name_inner]
                    chunks_outer = metrics_analysis_results.chunks_author_collection[author_name][collection_name_outer]
                    top_function_words_names, pca_data = self._get_pca_data(chunks_inner + chunks_outer)

                    collection_vs_collection_per_author_analysis[author_name][collection_name_outer][collection_name_inner] = PCAAnalysis._fit_transform(pca_data, top_function_words_names)

        return collection_vs_collection_per_author_analysis

    def _fit_transform_collections_per_author_chunks(self, metrics_analysis_results: MetricsAnalysisResults) -> Dict[str, PCAAnalysisData]:
        collections_per_author_analysis = {}

        for author_name in metrics_analysis_results.author_names:
            chunks_metrics = []
            for collection_name in metrics_analysis_results.collection_names:
                chunks_metrics.extend(metrics_analysis_results.chunks_author_collection[author_name][collection_name])
            top_function_words_names, pca_data = self._get_pca_data(chunks_metrics)
            collections_per_author_analysis[author_name] = PCAAnalysis._fit_transform(pca_data, top_function_words_names)
            
        return collections_per_author_analysis
    
    def _fit_transform_all_chunks(self, metrics_analysis_results: MetricsAnalysisResults) -> PCAAnalysisData:
        chunks_metrics = []
        for author_name in metrics_analysis_results.author_names:
            for collection_name in metrics_analysis_results.collection_names:
                chunks_metrics.extend(metrics_analysis_results.chunks_author_collection[author_name][collection_name])

        top_function_words_names, pca_data = self._get_pca_data(chunks_metrics)
        return PCAAnalysis._fit_transform(pca_data, top_function_words_names)
    
    def _fit_transform_author_collection_chunks(self, metrics_analysis_results: MetricsAnalysisResults) -> Dict[str, Dict[str, PCAAnalysisData]]:
        author_collection_analysis = {}

        for author_name in metrics_analysis_results.author_names:
            author_collection_analysis[author_name] = {}
            for collection_name in metrics_analysis_results.collection_names:
                chunks_metrics = metrics_analysis_results.chunks_author_collection[author_name][collection_name]
                top_function_words_names, pca_data = self._get_pca_data(chunks_metrics)
                author_collection_analysis[author_name][collection_name] = PCAAnalysis._fit_transform(pca_data, top_function_words_names)
            
        return author_collection_analysis
    
    def _transform_all_chunks(self, metrics_analysis_results: MetricsAnalysisResults, pca_analysis_data: PCAAnalysisData) -> PCAAnalysisData:
        chunks_metrics = []
        for author_name in metrics_analysis_results.author_names:
            for collection_name in metrics_analysis_results.collection_names:
                chunks_metrics.extend(metrics_analysis_results.chunks_author_collection[author_name][collection_name])

        _, pca_data = self._get_pca_data(
            metrics_data=chunks_metrics, 
            top_function_words_names=pca_analysis_data.top_function_words
        )
        return PCAAnalysis._transform(
            pca_data=pca_data, 
            pca_analysis_data=pca_analysis_data
        )
    
    def _get_pca_data(self, metrics_data: List[MetricData], top_function_words_names: List[str] = None) -> pd.DataFrame:
        """Get the PCA of the analysis data"""
        processed_columns = [f.name for f in fields(MetricData)]
        processed_columns.remove("sorted_function_words")
        processed_columns.remove("punctuation_frequency")
        punctuation_columns = list(punctuation) 
        top_function_words_names = top_function_words_names if top_function_words_names else self._get_top_function_words_names(metrics_data)
        all_columns = processed_columns + punctuation_columns + top_function_words_names
        df = pd.DataFrame([], columns=all_columns)

        for metric_data in metrics_data:
            serie = [getattr(metric_data, column) for column in processed_columns]
            serie.extend([metric_data.punctuation_frequency[column] for column in punctuation_columns])
            serie.extend([metric_data.sorted_function_words.get(column, 0)for column in top_function_words_names])
            df.loc[len(df)] = serie
        return top_function_words_names, df

    def _get_top_function_words_names(self, metrics_data: List[MetricData]) -> List[str]:
        """Get the top n function words from all the collections"""
        top_function_words_names = []
        for metric_data in metrics_data:
            top_n_function_words = list(metric_data.sorted_function_words.keys())[:self.configuration.top_n_function_words]
            top_function_words_names.extend(top_n_function_words)
        return list(set(top_function_words_names))

    @staticmethod
    def _pca_already_analysed(pca_collection_vs_collection, collection_name_outer: str, collection_name_inner: str) -> bool:
        return collection_name_inner in pca_collection_vs_collection and collection_name_outer in pca_collection_vs_collection[collection_name_inner]
    
    @staticmethod
    def _transform(pca_data: pd.DataFrame, pca_analysis_data: PCAAnalysisData) -> PCAAnalysisData:
        targets = ["source_name", "collection_name", "author_name"]
        features = [column for column in pca_data.columns if column not in targets]

        x = pca_data.loc[:, features].values
        x_scaled = pca_analysis_data.scaler.transform(x)
        principal_components = pca_analysis_data.pca.transform(x_scaled)
        pc_df = pd.DataFrame(data = principal_components, columns = ["PC1", "PC2"])
        pca_df = pd.concat([pc_df, pca_data[targets]], axis = 1)
        return PCAAnalysisData(
            pca=pca_analysis_data.pca,
            data=pca_data,
            results=pca_df,
            pc_variance=None,
            top_features=None,
            top_function_words=pca_analysis_data.top_function_words,
            scaler=pca_analysis_data.scaler
        )
    
    @staticmethod
    def _fit_transform(pca_data: pd.DataFrame, top_function_words: List[str]) -> PCAAnalysisData:
        """Get the PCA of the analysis data"""
        targets = ["source_name", "collection_name", "author_name"]
        features = [column for column in pca_data.columns if column not in targets]

        x = pca_data.loc[:, features].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x_scaled)
        pc_df = pd.DataFrame(data = principal_components, columns = ["PC1", "PC2"])
        pca_df = pd.concat([pc_df, pca_data[targets]], axis = 1) 

        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
        top_features = {
            'PC1': loadings['PC1'].abs().sort_values(ascending=False).index.tolist()[:PCAAnalysis.TOP_FEATURES],
            'PC2': loadings['PC2'].abs().sort_values(ascending=False).index.tolist()[:PCAAnalysis.TOP_FEATURES]
        }

        return PCAAnalysisData(
            pca=pca,
            data=pca_data,
            results=pca_df,
            pc_variance=pca.explained_variance_ratio_,
            top_features=top_features,
            top_function_words=top_function_words,
            scaler=scaler
        )
    