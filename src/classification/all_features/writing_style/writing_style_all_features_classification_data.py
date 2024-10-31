from dataclasses import dataclass

from src.classification.common.pca_classification_data import ClassificationData


@dataclass
class WritingStyleAllFeaturesClassificationResults:
    # Results of Classification performed on all chunks, all authors and collections are included in the pca.
    all_chunks_binary_classification: ClassificationData