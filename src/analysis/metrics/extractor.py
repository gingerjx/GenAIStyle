from dataclasses import fields
from typing import List

import pandas as pd
from src.analysis.metrics.models import MetricData, MetricsAnalysisResults
from src.settings import Settings


class FeatureExtractor:

    def __init__(self, settings: Settings, metrics_analysis_results: MetricsAnalysisResults) -> None:
        self.configuration = settings.configuration

        self.top_function_words_names = self._get_top_function_words_names(metrics_analysis_results.get_all_full_metrics())
        self.top_punctuations = self._get_top_punctuations(metrics_analysis_results.get_all_full_metrics())

    def get_features(self, metrics_data: List[MetricData]) -> pd.DataFrame:
        """Get the PCA of the analysis data"""
        processed_columns = [f.name for f in fields(MetricData)]
        processed_columns.remove("sorted_function_words")
        processed_columns.remove("punctuation_frequency")
        all_columns = processed_columns + self.top_punctuations + self.top_function_words_names
        df = pd.DataFrame([], columns=all_columns)

        for metric_data in metrics_data:
            serie = [getattr(metric_data, column) for column in processed_columns]
            serie.extend([metric_data.punctuation_frequency[column] for column in self.top_punctuations])
            serie.extend([metric_data.sorted_function_words.get(column, 0)for column in self.top_function_words_names])
            df.loc[len(df)] = serie
        return df

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
            top_n_punctuation = list(metric_data.punctuation_frequency.keys())[:self.configuration.top_n_punctuation]
            punctuation_names.extend(top_n_punctuation)
        return list(set(punctuation_names))