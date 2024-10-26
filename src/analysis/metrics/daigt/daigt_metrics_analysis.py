from src.analysis.metrics.common.metrics_analysis import MetricsAnalysis
from src.analysis.metrics.common.metrics_data import MetricData
from src.analysis.metrics.daigt.daigt_metrics_data import DaigtMetricsAnalysisResults
from src.analysis.preprocessing.daigt.daigt_preprocessing_data import DaigtPreprocessingResults


class DaigtMetricsAnalysis(MetricsAnalysis):

    def analyze(self, preprocessing_results: DaigtPreprocessingResults) -> DaigtMetricsAnalysisResults:
        """Analyze the authors and their collections"""
        metrics_analysis_results = DaigtMetricsAnalysisResults(
            collection_names=preprocessing_results.collection_names
        )

        for collection_name in preprocessing_results.collection_names:
            for preprocessing_chunk_data in preprocessing_results.chunks[collection_name]:
                metrics_analysis = MetricData(
                    author_name=None,
                    collection_name=collection_name,
                    **MetricsAnalysis._analyze(preprocessing_chunk_data)
                )
                metrics_analysis_results.chunks_collection[collection_name].append(metrics_analysis)
                
            preprocessing_data = preprocessing_results.full[collection_name]
            metrics_analysis = MetricData(
                author_name=None,
                collection_name=collection_name,
                **MetricsAnalysis._analyze(preprocessing_data)
            )
            metrics_analysis_results.full_collection[collection_name] = metrics_analysis

        return metrics_analysis_results