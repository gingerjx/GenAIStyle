from typing import Dict, List
import json

from src.analysis.analysis_data import AnalysisData, MetricData, Metadata
from src.analysis.metadata_analysis import MetadataAnalysis
from src.analysis.metrics_analysis import MetricsAnalysis
from src.analysis.pca_analysis import PCAAnalysis
from src.analysis.preprocessing_data import PreprocessingData
from src.file_utils import FileUtils
from src.models.author import Author
from src.settings import Settings
import jsonpickle

class Analysis():

    def __init__(self, 
                 settings: Settings, 
                 preprocessing_data: PreprocessingData,
                 read_from_file: bool = False
            ) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.preprocessing_data = preprocessing_data
        self.read_from_file = read_from_file
    
    def get_analysis(self, authors: List[Author]) -> AnalysisData:    
        if self.read_from_file:
            return FileUtils.read_analysis_data(self.paths.analysis_filepath)
        return self.analyze(authors)
    
    def analyze(self, authors: List[Author]) -> AnalysisData:
        """Analyze the authors and their collections"""
        analysis_data = AnalysisData(
            author_names=[author.name for author in authors],
            collection_names=[collection.name for collection in authors[0].cleaned_collections],
            metadata=Metadata(percentage_of_removed_text=MetadataAnalysis.get_percentage_of_removed_text(authors))
        )

        for author in authors:
            for collection in author.cleaned_collections:
                analysis_results = MetricsAnalysis._analyze(self.preprocessing_data[author][collection])
                metrics = MetricData(
                    author_name=author.name, 
                    collection_name=collection.name, 
                    **analysis_results
                )
                analysis_data.collection_author_metrics[collection.name][author.name] = metrics
                analysis_data.author_collection_metrics[author.name][collection.name] = metrics

        analysis_data.metadata.cross_top_function_words_names = MetadataAnalysis.get_cross_top_function_words_names(analysis_data, self.configuration.top_n_function_words)
        analysis_data.pca = PCAAnalysis.get_analysis(analysis_data)
        
        self._save_analysis_data(analysis_data)

        return analysis_data
    
    def _save_analysis_data(self, data: Dict[str, List[AnalysisData]]) -> None:
        """Save the analysis data to a file"""
        self.paths.analysis_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths.analysis_filepath, 'w', encoding='utf-8') as f:
            json_data = jsonpickle.encode(data)
            json.dump(json_data, f, indent=4)

