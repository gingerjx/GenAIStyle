from src.analysis.entropy.writing_style.writing_style_entropy_data import EntropyResults
from src.analysis.preprocessing.wiriting_style.writing_style_preprocessing_data import WritingStylePreprocessingResults
from src.analysis.visualization.analysis_visualization import AnalysisVisualization
from dash import dcc, html, Dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np

class WritingStyleEntropyAnalysisVisualizationDashApp(AnalysisVisualization):

    def __init__(self, 
                 preprocessing_results: WritingStylePreprocessingResults,
                 entropy_analysis_results: EntropyResults
    ) -> None:
        self.preprocessing_results = None # Change it!!!
        self.entropy_analysis_results = entropy_analysis_results
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Dropdown(
                id='features-dropdown',
                options=[
                    {'label': 'Feature 1', 'value': 'feature1'},
                    {'label': 'Feature 2', 'value': 'feature2'},
                ],
                value='feature1'
            ),
            html.Div(
                id='chunks_sequence_entropy', 
                style={"background-color": "white"}
            ),
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('chunks_sequence_entropy', 'children'),
            Input('features-dropdown', 'value'),
        )
        def chunks_sequence_entropy(feature: str) -> html.Div:
            words_elements = []
            all_chunks = self.preprocessing_results.get_all_chunks_preprocessing_data()

            for chunk_id, chunk_sequence in self.entropy_analysis_results.all_chunks_sequence_entropy.items():
                chunk = [chunk for chunk in all_chunks if chunk_id == chunk.chunk_id][0]
                for token, token_repetition in zip(chunk.split[:500], chunk_sequence.match_lengths[:500]):
                    words_elements.append(html.Span(token + " ", style={"background-color": "white"}))

            return words_elements
        
    def run(self, port: int):
        # self.app.run(
        #     port=port,
        #     jupyter_height=800,
        #     debug=False,
        # )  
        import numpy as np
        import pandas as pd

        collection_names = ["books", "gpt-3.5-turbo-0125", "gpt-4o", "gemini-1.5-flash", "open-mixtral-8x7b", "claude-3-haiku-20240307"]
        df = pd.DataFrame(columns=collection_names)

        sequence_entropies = list(self.entropy_analysis_results.all_chunks_sequence_entropy.values())
        feaure_entropies = list(self.entropy_analysis_results.all_chunks_features_entropy.values())

        for i, collection_name in enumerate(collection_names):
            entropies_per_collection = len(sequence_entropies) // 6
            left_range = i * entropies_per_collection
            right_range = (i + 1) * entropies_per_collection
            se = np.array([entropy.total_entropy for entropy in sequence_entropies[left_range:right_range]])
            fe = np.array([entropy.total_entropy for entropy in feaure_entropies[left_range:right_range]])
            df[collection_name] = pd.Series(data=[se.mean(), fe.mean()], index=["sequence_entropy", "feature_entropy"])

        return df


class WritingStyleEntropyAnalysisVisualization(AnalysisVisualization):
    
    def __init__(self, entropy_analysis_results: EntropyResults):
        self.entropy_analysis_results = entropy_analysis_results

    def visualize(self):
        self._visualize_heatmap()

    def _visualize_heatmap(self):
        self._find_average_samples() 

    def _find_average_samples(self):
        collection_name = "books"
        feature_entropies = self.entropy_analysis_results.all_chunks_features_entropy[collection_name].values()
        features_entropies_values = [list(entropy.features_entropy.values()) for entropy in feature_entropies]
        features_entropies_values = np.array(features_entropies_values)
        
        means = np.mean(features_entropies_values, axis=0)
        std_devs = np.std(features_entropies_values, axis=0, ddof=1)  # Sample standard deviation (ddof=1)
        std_errors = std_devs / np.sqrt(features_entropies_values.shape[0])  

        pass
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 5))
        # plt.bar([i for i in range(means.shape[0])], means, yerr=std_errors, capsize=5, alpha=0.7, color='skyblue')
        # plt.ylabel("Feature Values")
        # plt.title("Feature Averages with Uncertainty (Standard Error Bars)")
        # plt.show()
