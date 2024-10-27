from dataclasses import dataclass

from sklearn.decomposition import PCA

from src.analysis.pca.common.pca_data import PCAAnalysisData


@dataclass
class DaigtPCAResults:
    all_chunks: PCAAnalysisData