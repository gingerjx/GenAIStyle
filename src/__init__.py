from src.file_utils import FileUtils
from src.datasets.writing_style.author import Author
from src.datasets.common.collections.collection import Collection

from src.settings import Settings
from src.generation.text_generator import TextGenerator
from src.generation.generated_text import GeneratedText
from src.datasets.common.cleaner import Cleaner
from src.analysis.metadata.metadata_analysis import MetadataAnalysis
from src.analysis.metadata.daigt.daigt_metadata_analysis import DaigtMetadataAnalysis
from src.analysis.metrics.common.metrics_analysis import MetricsAnalysis
from src.analysis.pca.writing_style.writing_style_pca_analysis import WritingStylePCAAnalysis
from src.analysis.pca.writing_style.visualization.writing_style_pca_analysis_visualization import WritingStylePCAAnalysisResults
from src.analysis.metrics.writing_style.visualization.writing_style_metrics_analysis_visualization import WritingStyleMetricsAnalysisVisualization
from src.classification.classification import *
from src.classification.classification_data import ClassificationResultsTransformer
from src.classification.classification_visualization import ClassificationVisualization
from src.analysis.feature.common.feature_extractor import FeatureExtractor

from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.datasets.writing_style.writing_style_dataset import WritingStyleDataset
from src.analysis.metadata.writing_style.writing_style_metadata_analysis import WritingStyleMetadataAnalysis
from src.analysis.preprocessing.wiriting_style.wiriting_style_preprocessing import WritingStylePreprocessing
from src.analysis.preprocessing.daigt.daigt_preprocessing import DaigtPreprocessing
from src.analysis.metrics.daigt.daigt_metrics_analysis import DaigtMetricsAnalysis
from src.analysis.metrics.writing_style.writing_style_metrics_analysis import WritingStyleMetricsAnalysis
from src.analysis.pca.writing_style.visualization.writing_style_pca_analysis_visualization import WritingStylePCAAnalysisVisualization