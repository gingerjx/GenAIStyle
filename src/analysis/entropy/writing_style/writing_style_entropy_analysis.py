from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from src.analysis.entropy.writing_style.writing_style_entropy_data import *
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.common.metrics_data import MetricData
from src.analysis.metrics.writing_style.writing_style_metrics_data import WritingStyleMetricsAnalysisResults
from src.analysis.preprocessing.common.preprocessing_data import PreprocessingData
from src.analysis.preprocessing.wiriting_style.writing_style_preprocessing_data import WritingStylePreprocessingResults
from src.settings import Settings
from pandas._libs.interval import Interval

class EntropySequenceAnalysis:

    def __init__(self, words: List[str]) -> None:
        self.words = [word.lower() for word in words]
        self.N = len(self.words)
        self.words_dict = {word: [] for word in set(self.words)}

    def analyze(self) -> ChunkSequenceEntropyData:
        match_lengths = []

        for current_start in range(self.N):
            word = self.words[current_start]

            if match_lengths and match_lengths[-1] > 1:
                match_lengths.append(match_lengths[-1] - 1)
                self.words_dict[word].append(current_start)
                continue

            if len(self.words_dict[word]) > 0:
                match_lengths.append(self._find_shortest_unique_subsequence(current_start=current_start))
            else:
                match_lengths.append(1)

            self.words_dict[word].append(current_start)

        return ChunkSequenceEntropyData(
            entropy=self.N * np.log2(self.N) / sum(match_lengths),
            match_lengths=match_lengths
        )

    def _find_shortest_unique_subsequence(self, current_start: int) -> int:
        word = self.words[current_start]
        indices = self.words_dict[word]
        results = []

        for prev_start in indices:
            results.append(self.__find_shortest_unique_subsequence(
                    prev_start=prev_start, 
                    current_start=current_start
                )
            )

        return max(results)

    def __find_shortest_unique_subsequence(self, prev_start: int, current_start: int) -> int:
        shift = 1
        while self.N >= current_start + shift:
            if self.words[prev_start : prev_start+shift] == self.words[current_start : current_start+shift]:
                shift += 1
            else:
                return shift
        return shift
    
class WritingStyleEntropyAnalysis:
    
    def __init__(self, settings: Settings, feature_extractor: FeatureExtractor) -> None:
        self.configuration = settings.configuration
        self.feature_extractor = feature_extractor

    def analyze(self, 
        preprocessing_results: WritingStylePreprocessingResults,
        metrics_analysis_results: WritingStyleMetricsAnalysisResults, 
    ) -> EntropyResults:
        entropy_results = EntropyResults(
            collection_names=preprocessing_results.collection_names
        )

        entropy_results.features_distributions = self._get_features_distributions(metrics_analysis_results)
        entropy_results = self._get_chunks_features_entropies(
            metrics_analysis_results=metrics_analysis_results,
            entropy_results=entropy_results
        )
        entropy_results = self._get_chunks_sequence_entropy(
            preprocessing_results=preprocessing_results,
            entropy_results=entropy_results
        )
        self._calculate_entropies_average_data(entropy_results)
        
        return entropy_results
    
    def _get_features_distributions(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults) -> Dict[str, FeatureDistributionData]:
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

    def _get_chunks_features_entropies(self, metrics_analysis_results: WritingStyleMetricsAnalysisResults, entropy_results: EntropyResults) -> EntropyResults:
        for chunk_metrics in metrics_analysis_results.get_all_chunks_metrics():
            feature_names = self.feature_extractor.get_feature_names_without_metadata()
            chunk_features_entropy_data = ChunkFeaturesEntropyData()

            for feature_name in feature_names:
                feature_distribution = entropy_results.features_distributions[feature_name]
                field_extractor = self._get_field_extractor(feature_name)
                feature_value = field_extractor(chunk_metrics, feature_name)
                bin = self._find_bin(feature_value, feature_distribution.bins)
                chunk_features_entropy_data.features_entropy[feature_name] = self._calculate_entropy(bin.probability)
            
            entropy_results.collections_entropies[chunk_metrics.collection_name] \
                .chunks_features_entropies[chunk_metrics.chunk_id] = chunk_features_entropy_data

        return entropy_results
             
    def _find_bin(self, value: Any, bins: List[BinData]) -> BinData:
        for bin in bins:
            if value in bin.interval:
                return bin
        return None

    def _calculate_entropy(self, probability: float) -> float:
        return -np.log2(probability)
    
    def _get_chunks_sequence_entropy(self, preprocessing_results: WritingStylePreprocessingResults, entropy_results: EntropyResults) -> EntropyResults:   
        for chunk_metrics in preprocessing_results.get_all_chunks_preprocessing_data():
            sequence_entropy = EntropySequenceAnalysis(chunk_metrics.words).analyze()
            entropy_results.collections_entropies[chunk_metrics.collection_name] \
                .chunks_sequence_entropy[chunk_metrics.chunk_id] = sequence_entropy

        return entropy_results
    
    def _calculate_entropies_average_data(self, entropy_results: EntropyResults) -> EntropyResults:
        feature_names = self.feature_extractor.get_feature_names_without_metadata()
        for collection_names in entropy_results.collection_names:
            collection_entropies = entropy_results.collections_entropies[collection_names] 
            collection_entropies_list = [
                list(chunk_features_entropies.features_entropy.values()) 
                for chunk_features_entropies in collection_entropies.chunks_features_entropies.values()
            ]
            collection_entropies_list = np.array(collection_entropies_list)

            means = np.mean(collection_entropies_list, axis=0)
            std_devs = np.std(collection_entropies_list, axis=0, ddof=1)
            std_errors = std_devs / np.sqrt(collection_entropies_list.shape[0])  

            for i, feature_name in enumerate(feature_names):
                feature_entropy_average_data = CollectionEntropyAverageData(
                    average=means[i],
                    average_uncertainty=std_errors[i],
                    std=std_devs[i]
                )
                collection_entropies.average_data[feature_name] = feature_entropy_average_data
        
        for collection_name in entropy_results.collection_names:
            collection_entropies = entropy_results.collections_entropies[collection_name]
            sequence_entropies = [entropy.entropy for entropy in collection_entropies.chunks_sequence_entropy.values()]
            sequence_entropies = np.array(sequence_entropies)

            average = np.mean(sequence_entropies)
            std_dev = np.std(sequence_entropies, ddof=1)
            std_error = std_dev / np.sqrt(sequence_entropies.shape[0])

            collection_entropies.average_data["sequence"] = CollectionEntropyAverageData(
                average=average,
                average_uncertainty=std_error,
                std=std_dev
            )

        return entropy_results