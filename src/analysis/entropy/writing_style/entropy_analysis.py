from typing import Callable, List

import numpy as np
import pandas as pd
from src.analysis.entropy.writing_style.entropy_data import BinData, EntropyData, EntropyFeatureData, FeatureDistributionData
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.common.metrics_data import MetricData
from src.analysis.metrics.writing_style.writing_style_metrics_data import WritingStyleMetricsAnalysisResults
from src.settings import Settings
from pandas._libs.interval import Interval

class EntropyAnalysis:
    
    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        self.configuration = settings.configuration
        self.feature_extractor = feature_extractor

    def analyze(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> None:
        entropy_data = self._get_entropy_data(metrics_analysis_results)
        pass

    def _get_entropy_data(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> EntropyData:
        all_chunks = metrics_analysis_results.get_all_chunks_metrics()    
        feature_names = self.feature_extractor.get_feature_names_without_metadata()
        entropy_data = EntropyData(distributions={})
        
        for feature_name in feature_names:
            features = self._get_features(feature_name, all_chunks)
            features_values = [f.value for f in features]
            total_count = len(features_values)

            min_bin = min(features_values)
            max_bin = max(features_values) + 0.01
            bins = np.linspace(min_bin, max_bin, self.configuration.ws_entropy_analysis_number_of_bins + 1)
            binned_features = np.digitize(features_values, bins=bins)
            
            feature_distribution_data = FeatureDistributionData(
                min_value=min_bin,
                max_value=max_bin,
                size=self.configuration.ws_entropy_analysis_number_of_bins,
                bins=[]
            )

            for i in range(self.configuration.ws_entropy_analysis_number_of_bins):
                interval = Interval(
                    left=bins[i], 
                    right=bins[i + 1],
                    closed='left',
                )
                feature_distribution_data.bins.append(
                    BinData(
                        interval=interval,
                        index=i,
                        count=0,
                        probability=0,
                        features=[],
                    )
                )

            for feature, index in zip(features, binned_features):
                feature_distribution_data.bins[index - 1].count += 1
                feature_distribution_data.bins[index - 1].features.append(feature)

            for bin in feature_distribution_data.bins:
                bin.probability = bin.count / total_count
            
            entropy_data.distributions[feature_name] = feature_distribution_data

        return entropy_data

    def _get_features(self, feature_name: str, chunks: List[MetricData]) -> List[EntropyFeatureData]:
            field_extractor = self._get_field_extractor(feature_name)
            return [
                EntropyFeatureData(
                    name=feature_name,
                    value=field_extractor(m, feature_name),
                    collection_name=m.collection_name,
                    author_name=m.author_name,
                    source_name=m.source_name
                ) for m in chunks
            ]

    def _get_field_extractor(self, feature_name: str) -> Callable[[str], float]:
        if feature_name in self.feature_extractor.get_top_punctuation_features():
            return lambda metrics, feature: metrics.punctuation_frequency.get(feature, 0)
        elif feature_name in self.feature_extractor.get_top_function_words_features():
            return lambda metrics, feature: metrics.sorted_function_words.get(feature, 0)
        else:
            return lambda metrics, feature: getattr(metrics, feature)

