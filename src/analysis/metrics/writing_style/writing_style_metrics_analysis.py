from src.analysis.metrics.common.metrics_analysis import MetricsAnalysis
from src.analysis.metrics.common.metrics_data import MetricData
from src.analysis.metrics.writing_style.writing_style_metrics_data import WritingStyleMetricsAnalysisResults
from src.analysis.preprocessing.wiriting_style.writing_style_preprocessing_data import WritingStylePreprocessingResults


class WritingStyleMetricsAnalysis(MetricsAnalysis):    

    def analyze(self, preprocessing_results: WritingStylePreprocessingResults) -> WritingStyleMetricsAnalysisResults:
        """Analyze the authors and their collections"""
        metrics_analysis_results = WritingStyleMetricsAnalysisResults(
            author_names=preprocessing_results.author_names,
            collection_names=preprocessing_results.collection_names
        )

        for author_name in preprocessing_results.author_names:
            for collection_name in preprocessing_results.collection_names:
                for preprocessing_chunk_data in preprocessing_results.chunks[author_name][collection_name]:
                    metrics_analysis = MetricData(
                        chunk_id=preprocessing_chunk_data.chunk_id,
                        author_name=author_name,
                        collection_name=collection_name,
                        **MetricsAnalysis._analyze(preprocessing_chunk_data)
                    )
                    metrics_analysis_results.chunks_author_collection[author_name][collection_name].append(metrics_analysis)
                    metrics_analysis_results.chunks_collection_author[collection_name][author_name].append(metrics_analysis)
                    
                preprocessing_data = preprocessing_results.full[author_name][collection_name]
                metrics_analysis = MetricData(
                    chunk_id=preprocessing_chunk_data.chunk_id,
                    author_name=author_name,
                    collection_name=collection_name,
                    **MetricsAnalysis._analyze(preprocessing_data)
                )
                metrics_analysis_results.full_author_collection[author_name][collection_name] = metrics_analysis
                metrics_analysis_results.full_collection_author[collection_name][author_name] = metrics_analysis

        return metrics_analysis_results