from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from src.analysis.entropy.writing_style.entropy_data import BinData, ChunkFeatureEntropyData, EntropyResults, FeatureData, FeatureDistributionData
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.common.metrics_data import MetricData
from src.analysis.metrics.writing_style.writing_style_metrics_data import WritingStyleMetricsAnalysisResults
from src.analysis.preprocessing.common.preprocessing_data import PreprocessingData
from src.settings import Settings
from pandas._libs.interval import Interval

class EntropyAnalysis:
    
    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        self.configuration = settings.configuration
        self.feature_extractor = feature_extractor

    def analyze(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> None:
        distributions = self._get_entropy_data(metrics_analysis_results)
        all_chunks_features_entropy = self._get_chunks_features_entropy(
            metrics_analysis_results=metrics_analysis_results,
            distributions=distributions
        )
        sequence_entropy = self._calculate_sequence_entropy(["To", "be", "or", "not", "to", "be", "there", "to", "make", "it", "true" ])
        pass
    
    def _get_entropy_data(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> Dict[str, FeatureDistributionData]:
        all_chunks = metrics_analysis_results.get_all_chunks_metrics()    
        feature_names = self.feature_extractor.get_feature_names_without_metadata()
        distributions = {}
        
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
            
            distributions[feature_name] = feature_distribution_data

        return distributions

    def _get_features(self, feature_name: str, chunks: List[MetricData]) -> List[FeatureData]:
            field_extractor = self._get_field_extractor(feature_name)
            return [
                FeatureData(
                    name=feature_name,
                    value=field_extractor(m, feature_name),
                    collection_name=m.collection_name,
                    author_name=m.author_name,
                    source_name=m.source_name
                ) for m in chunks
            ]

    def _get_field_extractor(self, feature_name: str) -> Callable[[str], float]:
        if feature_name in self.feature_extractor.get_top_punctuation_features():
            return lambda metrics, feature_name: metrics.punctuation_frequency.get(feature_name, 0)
        elif feature_name in self.feature_extractor.get_top_function_words_features():
            return lambda metrics, feature_name: metrics.sorted_function_words.get(feature_name, 0)
        else:
            return lambda metrics, feature_name: getattr(metrics, feature_name)

    def _get_chunks_features_entropy(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults, distributions: Dict[str, FeatureDistributionData]) -> Dict[MetricData, ChunkFeatureEntropyData]:
        all_chunks_features_entropy = {}
        
        for chunk_metrics in metrics_analysis_results.get_all_chunks_metrics():
            feature_names = self.feature_extractor.get_feature_names_without_metadata()
            entropy_values = {}

            for feature_name in feature_names:
                feature_distribution = distributions[feature_name]
                field_extractor = self._get_field_extractor(feature_name)
                feature_value = field_extractor(chunk_metrics, feature_name)
                bin = self._find_bin(feature_value, feature_distribution.bins)
                entropy_values[feature_name] = self._calculate_entropy(bin.probability)
            
            all_chunks_features_entropy[chunk_metrics] = ChunkFeatureEntropyData(
                total_entropy=sum(entropy_values.values()),
                features_entropy=entropy_values
            )

        return all_chunks_features_entropy
             
    def _find_bin(self, value: Any, bins: List[BinData]) -> BinData:
        for bin in bins:
            if value in bin.interval:
                return bin
        return None

    def _calculate_entropy(self, probability: float) -> float:
        return -np.log2(probability)

    def _calculate_sequence_entropy(self, words: List[str]) -> float:
        N = len(words)
        
        # Step 2: Calculate match lengths (Λ_i)
        match_lengths = []
        for i in range(N):
            # Start with the shortest subsequence length of 1
            match_length = 1
            while i + match_length <= N:
                # Subsequence starting at current position of given length
                subseq = words[i:i + match_length]
                
                # Check if this subsequence has appeared before this position
                found = False
                for j in range(i):
                    if words[j:j + match_length] == subseq:
                        found = True
                        break
                
                # If subsequence was not found before, we have our match length
                if not found:
                    match_lengths.append(match_length)
                    break
                
                # Otherwise, increase the subsequence length and try again
                match_length += 1

        # Step 3: Calculate entropy using the formula
        entropy = (N * np.log2(N)) / sum(match_lengths)
        
        return entropy