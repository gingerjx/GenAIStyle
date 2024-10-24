from src.file_utils import FileUtils
from src.models.author import Author
from src.datasets.common.collections.collection import Collection
from src.analysis.preprocessing.analysis import Preprocessing
from src.settings import Settings
from src.generation.text_generator import TextGenerator
from src.generation.generated_text import GeneratedText
from src.datasets.common.cleaner import Cleaner
from src.analysis.metadata.metadata_analysis import MetadataAnalysis
from src.analysis.metadata.daigt.daigt_metadata_analysis import DaigtMetadataAnalysis
from src.analysis.metrics.analysis import MetricsAnalysis
from src.analysis.pca.analysis import PCAAnalysis
from src.analysis.visualization.pca_analysis_visualization import PCAAnalysisVisualization
from src.analysis.visualization.metrics_analysis_visualization import MetricsAnalysisVisualization
from src.classification.classification import *
from src.classification.classification_data import ClassificationResultsTransformer
from src.classification.classification_visualization import ClassificationVisualization
from src.analysis.metrics.extractor import FeatureExtractor

from src.datasets.daigt.daigt_dataset import DaigtDataset
from src.datasets.writing_style.writing_style_dataset import WritingStyleDataset
from src.analysis.metadata.writing_style.writing_style_metadata_analysis import WritingStyleMetadataAnalysis