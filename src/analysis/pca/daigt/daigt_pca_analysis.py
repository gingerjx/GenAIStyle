
import pandas as pd
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.metrics.daigt.daigt_metrics_data import DaigtMetricsAnalysisResults
from src.analysis.pca.common.pca_data import PCAAnalysisData
from src.analysis.pca.daigt.daigt_pca_data import DaigtPCAResults
from src.analysis.pca.writing_style.writing_style_pca_data import WritingStylePCAAnalysisResults
from src.settings import Settings


class DaigtPCAAnalysis:
    
    def __init__(self, 
                 settings: Settings, 
                 writing_style_feature_extractor: FeatureExtractor,
                 writing_style_pca_analysis_results: WritingStylePCAAnalysisResults) -> None:
        self.configuration = settings.configuration
        self.writing_style_feature_extractor = writing_style_feature_extractor
        self.writing_style_pca_analysis_results = writing_style_pca_analysis_results

    def analyze(self, metrics_analysis_results: DaigtMetricsAnalysisResults) -> None:
        chunks_metrics = metrics_analysis_results.get_all_chunks_metrics()
        pca_data = self.writing_style_feature_extractor.get_features(chunks_metrics)
        pca, scaler, pca_df = self._transform(pca_data)

        return DaigtPCAResults(
            all_chunks=PCAAnalysisData(
                pca=pca,
                data=pca_data,
                results=pca_df,
                scaler=scaler
            )
        )

    def _transform(self, pca_data: pd.DataFrame):
        targets = ["source_name", "collection_name", "author_name"]
        features = [column for column in pca_data.columns if column not in targets]
        targets.remove("author_name")

        scaler = self.writing_style_pca_analysis_results.all_chunks.scaler
        pca = self.writing_style_pca_analysis_results.all_chunks.pca

        x = pca_data.loc[:, features].values
        x_scaled = scaler.transform(x)

        principal_components = pca.transform(x_scaled)
        pc_df = pd.DataFrame(data = principal_components, columns = ["PC1", "PC2"])
        pca_df = pd.concat([pc_df, pca_data[targets]], axis = 1) 

        return pca, scaler, pca_df