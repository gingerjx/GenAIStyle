from typing import Dict, List
import json

from src.analysis.analysis_data import AnalysisData, AnalysisResults, MetricData, Metadata
from src.analysis.metadata_analysis import MetadataAnalysis
from src.analysis.metrics_analysis import MetricsAnalysis
from src.analysis.pca_analysis import PCAAnalysis
from src.analysis.preprocessing_data import PreprocessingData, PreprocessingResults
from src.file_utils import FileUtils
from src.models.author import Author
from src.settings import Settings
import jsonpickle

class Analysis():

    def __init__(self, 
                 settings: Settings, 
                 preprocessing_results: PreprocessingResults,
                 read_from_file: bool = False
            ) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.preprocessing_results = preprocessing_results
        self.read_from_file = read_from_file
    
    def get_analysis(self, authors: List[Author]) -> AnalysisResults:    
        if self.read_from_file:
            return FileUtils.read_analysis_data(self.paths.analysis_filepath)
        return self.analyze(authors)
    
    def analyze(self, authors: List[Author]) -> AnalysisResults:
        """Analyze the authors and their collections"""
        author_names=[author.name for author in authors]
        collection_names=[collection.name for collection in authors[0].cleaned_collections]
        analysis_results = AnalysisResults(
            full=self._get_analysis_data(author_names, collection_names),
            metadata=Metadata(percentage_of_removed_text=MetadataAnalysis.get_percentage_of_removed_text(authors))
        )          
       
        self._save_analysis_results(analysis_results)

        return analysis_results
    
    def _get_analysis_data(self, author_names: List[str], collection_names: List[str], chunk_id: int = None) -> AnalysisData:
        analysis_data = AnalysisData(
            author_names=author_names,
            collection_names=collection_names
        )
        for author_name in author_names:
            for collection_name in collection_names:
                metrics = MetricData(
                    author_name=author_name, 
                    collection_name=collection_name, 
                    **MetricsAnalysis._analyze(self._get_preprocessing_data(author_name, collection_name, chunk_id))
                )
                analysis_data.collection_author_metrics[collection_name][author_name] = metrics
                analysis_data.author_collection_metrics[author_name][collection_name] = metrics

        analysis_data.metadata.cross_top_function_words_names = MetadataAnalysis.get_cross_top_function_words_names(analysis_data, self.configuration.top_n_function_words)
        analysis_data.pca = PCAAnalysis.get_analysis(analysis_data)

        return analysis_data

    def _get_preprocessing_data(self, author_name: str, collection_name: str, chunk_id: int) -> PreprocessingData:
        if chunk_id:
            return getattr(self.preprocessing_results[author_name][collection_name], "chunks")[chunk_id]
        return getattr(self.preprocessing_results[author_name][collection_name], "full")
    
    def _save_analysis_results(self, data: Dict[str, List[AnalysisData]]) -> None:
        """Save the analysis data to a file"""
        self.paths.analysis_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths.analysis_filepath, 'w', encoding='utf-8') as f:
            json_data = jsonpickle.encode(data)
            json.dump(json_data, f, indent=4)

