from dataclasses import fields
from string import punctuation
from typing import List

import pandas as pd
from src.analysis.metrics.models import MetricData
from src.settings import Settings


class MetricsExtractor:

    def __init__(self, settings: Settings):
        self.configuration = settings.configuration

    def get_features(self, metrics_data: List[MetricData], top_function_words_names: List[str] = None) -> pd.DataFrame:
        """Get the PCA of the analysis data"""
        processed_columns = [f.name for f in fields(MetricData)]
        processed_columns.remove("sorted_function_words")
        processed_columns.remove("punctuation_frequency")
        punctuation_columns = list(punctuation) 
        top_function_words_names = top_function_words_names if top_function_words_names else self.get_top_function_words_names(metrics_data)
        all_columns = processed_columns + punctuation_columns + top_function_words_names
        df = pd.DataFrame([], columns=all_columns)

        for metric_data in metrics_data:
            serie = [getattr(metric_data, column) for column in processed_columns]
            serie.extend([metric_data.punctuation_frequency[column] for column in punctuation_columns])
            serie.extend([metric_data.sorted_function_words.get(column, 0)for column in top_function_words_names])
            df.loc[len(df)] = serie
        return df

    def get_top_function_words_names(self, metrics_data: List[MetricData]) -> List[str]:
        """Get the top n function words from all the collections"""
        top_function_words_names = []
        for metric_data in metrics_data:
            top_n_function_words = list(metric_data.sorted_function_words.keys())[:self.configuration.top_n_function_words]
            top_function_words_names.extend(top_n_function_words)
        return list(set(top_function_words_names))