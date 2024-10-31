import plotly.graph_objects as go

from src.analysis.pca.daigt.daigt_pca_data import DaigtPCAResults
from src.analysis.visualization.analysis_visualization import AnalysisVisualization

class DaigtPCAVisualization(AnalysisVisualization):

    COLLECTION_COLORS = {
        "human": "#3498db", 
        "gpt-3.5-turbo-0125": "#e74c3c",
        "gpt-4o": "#2ecc71", 
        "gemini-1.5-flash": "#f1c40f", 
        "open-mixtral-8x7b": "#9b59b6", 
        "claude-3-haiku-20240307": "#e67e22",
    }

    def __init__(self, pca_results: DaigtPCAResults):
        self.pca_results = pca_results

    def visualize(self):
        self._visualize_all_chunks(self.pca_results)

    def _visualize_all_chunks(self, pca_results: DaigtPCAResults):
        fig = go.Figure()
        color_dict = DaigtPCAVisualization.get_color_dict(pca_results.collection_names)

        for collection_name in self.pca_results.collection_names:
            results = pca_results.all_chunks.results
            mask = results['collection_name'] == collection_name
            fig.add_trace(go.Scatter(
                x=results.loc[mask, 'PC1'],
                y=results.loc[mask, 'PC2'],
                mode='markers',
                marker=dict(color=color_dict[collection_name]),
                name=collection_name,
            ))

        fig.update_layout(
            xaxis_title='PC1',
            yaxis_title='PC2',
            title='PCA Analysis of All Chunks',
            showlegend=True,
        )
        fig.show()