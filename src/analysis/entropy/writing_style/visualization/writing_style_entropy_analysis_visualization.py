from src.analysis.entropy.writing_style.writing_style_entropy_data import EntropyResults
from src.analysis.feature.common.feature_extractor import FeatureExtractor
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
    
    def __init__(self, feature_extractor: FeatureExtractor, entropy_analysis_results: EntropyResults):
        self.entropy_analysis_results = entropy_analysis_results
        self.feature_extractor = feature_extractor

    def visualize(self):
        self._visualize_heatmap()

    def _visualize_heatmap(self):
        self._visualize_average_chunks_entropies()
        # self._visualize_averages_uncertainty() 
        pass
    
    def _visualize_average_chunks_entropies(self):
        data = []

        for collection_name in self.entropy_analysis_results.collection_names:
            collections_entropies = self.entropy_analysis_results.collections_entropies[collection_name]
            average_chunk_id = collections_entropies.average_chunk_id
            collection_chunks_entropies = list(collections_entropies.chunks_features_entropies[average_chunk_id].features_entropy.values())
            collection_chunks_entropies.append(collections_entropies.chunks_sequence_entropy[average_chunk_id].entropy)
            data.append(collection_chunks_entropies)

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=self.feature_extractor.get_feature_names_without_metadata() + ["sequence"],
            y=self.entropy_analysis_results.collection_names,
            colorscale='Viridis',
        ))
        fig.update_layout(title='Heatmap of average chunks entropies', xaxis_title='Feature', yaxis_title='Collection')
        fig.show()

    def _visualize_averages_uncertainty(self):
        feautre_names = self.feature_extractor.get_feature_names_without_metadata()
        collection_entropies = self.entropy_analysis_results.collections_entropies["books"] 
        collection_entropies_list = [
            list(chunk_features_entropies.features_entropy.values()) 
            for chunk_features_entropies in collection_entropies.chunks_features_entropies.values()
        ]
        collection_entropies_list = np.transpose(np.array(collection_entropies_list))

        fig = go.Figure()
        for i, feauture_entropies in enumerate(collection_entropies_list):
            fig.add_trace(
                go.Box(
                    y=feauture_entropies, 
                    name=feautre_names[i], 
                    boxmean='sd'
                )
            )
        
        sequence_entropies = np.array([
            sequence_entropy.entropy 
            for sequence_entropy in collection_entropies.chunks_sequence_entropy.values()
        ])
        fig.add_trace(
            go.Box(
                y=sequence_entropies, 
                name="sequence", 
                boxmean='sd'
            )
        )
        
        fig.update_layout(title=f'Box Plot of books entropies', yaxis_title='Value', showlegend=False)
        fig.show()