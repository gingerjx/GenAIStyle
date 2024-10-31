from dataclasses import fields
from typing import List

import pandas as pd
from src.analysis.metrics.common.metrics_data import MetricData, MetricsAnalysisResults
from src.settings import Settings


class FeatureExtractor:

    def __init__(self, settings: Settings, metrics_analysis_results: MetricsAnalysisResults) -> None:
        self.configuration = settings.configuration

        self.processed_columns = self._get_processed_columns()
        self.top_function_words_names = self._get_top_function_words_names(metrics_analysis_results.get_all_full_metrics())
        self.top_punctuations = self._get_top_punctuations(metrics_analysis_results.get_all_full_metrics())
        self.feature_names = self.processed_columns + self.top_punctuations + self.top_function_words_names

    def get_feature_names(self) -> List[str]:
        return self.feature_names
    
    def get_feature_names_without_metadata(self) -> List[str]:
        copy = self.feature_names.copy()
        copy.remove("source_name")
        copy.remove("collection_name")
        copy.remove("author_name")
        return copy
    
    def get_top_punctuation_features(self) -> List[str]:
        return self.top_punctuations
    
    def get_top_function_words_features(self) -> List[str]:
        return self.top_function_words_names
    
    def get_features(self, metrics_data: List[MetricData]) -> pd.DataFrame:
        """Get the PCA of the analysis data"""
        features = {feature_name: [] for feature_name in self.feature_names}
        
        for metric_data in metrics_data:
            for column in self.processed_columns:
                features[column].append(getattr(metric_data, column))
            for column in self.top_punctuations:
                features[column].append(metric_data.punctuation_frequency.get(column, 0))
            for column in self.top_function_words_names:
                features[column].append(metric_data.sorted_function_words.get(column, 0))

        return pd.DataFrame(features)

    def _get_processed_columns(self) -> List[str]:
        """Get the columns that are not function words or punctuations"""
        processed_columns = [f.name for f in fields(MetricData)]
        processed_columns.remove("sorted_function_words")
        processed_columns.remove("punctuation_frequency")
        return processed_columns
    
    def _get_top_function_words_names(self, metrics_data: List[MetricData]) -> List[str]:
        """
        Get the top n function words from given `metrics_data`. From each metric data `self.configuration.top_n_function_words` function words are selected,
        merged together and returned as a list without duplicates.
        """
        top_function_words_names = []
        for metric_data in metrics_data:
            top_n_function_words = list(metric_data.sorted_function_words.keys())[:self.configuration.top_n_function_words]
            top_function_words_names.extend(top_n_function_words)
        return list(set(top_function_words_names))
    
    def _get_top_punctuations(self, metrics_data: List[MetricData]) -> List[str]:
        """
        Get the punctuation from given `metrics_data`. From each metric data punctuation is selected,
        merged together and returned as a list without duplicates.
        """
        punctuation_names = []
        for metric_data in metrics_data:
            sorted_punctuation = dict(sorted(metric_data.punctuation_frequency.items(), key=lambda item: item[1], reverse=True))
            top_n_punctuation = list(sorted_punctuation.keys())[:self.configuration.top_n_punctuation]
            punctuation_names.extend(top_n_punctuation)
        return list(set(punctuation_names))