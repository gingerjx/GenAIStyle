from typing import List
from src.analysis.entropy.writing_style.writing_style_entropy_data import EntropyResults
from src.analysis.feature.common.feature_extractor import FeatureExtractor
from src.analysis.preprocessing.wiriting_style.writing_style_preprocessing_data import WritingStylePreprocessingResults
from src.analysis.visualization.analysis_visualization import AnalysisVisualization
from dash import dcc, html, Dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
from IPython.display import display, Markdown, Latex, HTML
from matplotlib import pyplot as plt

class WritingStyleEntropyAnalysisVisualizationDashApp(AnalysisVisualization):

    class Helper():
        @staticmethod
        def get_collection_buttons(parent: "WritingStyleEntropyAnalysisVisualizationDashApp", id: str, init: int = 0) -> dcc.Dropdown:
            return dcc.Dropdown(
                id=id,
                options=[
                    {'label': collection_name, 'value': collection_name}
                    for collection_name in parent.entropy_analysis_results.collection_names
                ],
                value=parent.entropy_analysis_results.collection_names[init]
            )

    def __init__(self, 
                 preprocessing_results: WritingStylePreprocessingResults,
                 entropy_analysis_results: EntropyResults,
                 feature_extractor: FeatureExtractor,
                 max_tokens_to_display: int = 1000,
                 top_n_features_to_display: int = 10
    ) -> None:
        self.preprocessing_results = preprocessing_results
        self.entropy_analysis_results = entropy_analysis_results
        self.feature_extractor = feature_extractor
        self.max_tokens_to_display = max_tokens_to_display
        self.top_n_features_to_display = top_n_features_to_display

        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                self.Helper.get_collection_buttons(self, 'collection-averages-uncertainty-dropdown'),
                dcc.Graph(id='collection-averages-uncertainty'),
            ]),

            html.Div([
                html.Div([
                    self.Helper.get_collection_buttons(self, 'collection-sequence-entropy-dropdown-1'),
                    html.P(id="collection-sequence-entropy-score-1"),
                    html.Div(id='collection-sequence-entropy-1'),
                ], style={"padding": "10px"}),
                html.Hr(),
                html.Div([
                    self.Helper.get_collection_buttons(self, 'collection-sequence-entropy-dropdown-2', init=1),
                    html.P(id="collection-sequence-entropy-score-2"),
                    html.Div(id='collection-sequence-entropy-2'),
                ], style={"padding": "10px"}),
            ], style={
                "background-color": "white", 
                "width": "100%", 
                "display": "flex", 
                "flex-direction": "row", 
                "align-items": "center"
            }),

            html.Div([
                html.Div([
                    self.Helper.get_collection_buttons(self, 'collection-features-entropy-dropdown-1'),
                    html.Div(id='collection-features-entropy-1'),
                ], style={"padding": "10px"}),
                html.Hr(),
                html.Div([
                    self.Helper.get_collection_buttons(self, 'collection-features-entropy-dropdown-2', init=1),
                    html.Div(id='collection-features-entropy-2'),
                ], style={"padding": "10px"}),
            ], style={
                "background-color": "white", 
                "width": "100%", 
                "display": "flex", 
                "flex-direction": "row", 
                "align-items": "center"
            }),
        ])

    def setup_callbacks(self):
        
        # Graph 1

        @self.app.callback(
            Output('collection-averages-uncertainty', 'figure'),
            Input('collection-averages-uncertainty-dropdown', 'value'),
        )
        def collection_averages_uncertainty(collection_name: str) -> dcc.Graph:
            feautre_names = self.feature_extractor.get_feature_names_without_metadata()
            collection_entropies = self.entropy_analysis_results.collections_entropies[collection_name] 
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

            sequence_entropies = np.array([
                sequence_entropy.entropy 
                for sequence_entropy in collection_entropies.chunks_ws_words_entropy.values()
            ])
            fig.add_trace(
                go.Box(
                    y=sequence_entropies, 
                    name="local_word_distribution", 
                    boxmean='sd'
                )
            )

            sequence_entropies = np.array([
                sequence_entropy.entropy 
                for sequence_entropy in collection_entropies.chunks_all_words_entropy.values()
            ])
            fig.add_trace(
                go.Box(
                    y=sequence_entropies, 
                    name="global_word_distribution", 
                    boxmean='sd'
                )
            )

            fig.update_layout(title=f"Box Plots of {collection_name}' attributes IC", yaxis_title='Value', showlegend=False)
            return fig
        
        # Visualization 1
        
        @self.app.callback(
            Output('collection-sequence-entropy-1', 'children'),
            Input('collection-sequence-entropy-dropdown-1', 'value'),
        )
        def collection_sequence_entropy_1(collection_name: str) -> html.Div:
            return _collection_sequence_entropy(collection_name)

        @self.app.callback(
            Output('collection-sequence-entropy-score-1', 'children'),
            Input('collection-sequence-entropy-dropdown-1', 'value'),
        )
        def collection_sequence_entropy_score_2(collection_name: str) -> str:
            return _collection_sequence_entropy_score(collection_name)
        
        @self.app.callback(
            Output('collection-sequence-entropy-2', 'children'),
            Input('collection-sequence-entropy-dropdown-2', 'value'),
        )
        def collection_sequence_entropy_2(collection_name: str) -> html.Div:
            return _collection_sequence_entropy(collection_name)

        @self.app.callback(
            Output('collection-sequence-entropy-score-2', 'children'),
            Input('collection-sequence-entropy-dropdown-2', 'value'),
        )
        def collection_sequence_entropy_score_2(collection_name: str) -> str:
            return _collection_sequence_entropy_score(collection_name)
        
        def _collection_sequence_entropy_score(collection_name: str) -> str:
            # Calculate the entropy score based on the selected collection
            collection_entropies = self.entropy_analysis_results.collections_entropies[collection_name]
            average_chunk_id = collection_entropies.average_chunk_id
            entropy_score = collection_entropies.chunks_sequence_entropy[average_chunk_id].entropy
            return f"Displayed tokens: {self.max_tokens_to_display}. Sequence entropy Score: {entropy_score:.2f}"

        def _collection_sequence_entropy(collection_name: str) -> html.Div:
            words_elements = []
            all_chunks = self.preprocessing_results.get_all_chunks_preprocessing_data()
            collection_entropies = self.entropy_analysis_results.collections_entropies[collection_name]
            collection_average_chunk_id = collection_entropies.average_chunk_id
            collection_average_chunk = [chunk for chunk in all_chunks if collection_average_chunk_id == chunk.chunk_id][0]
            collection_average_sequence_entropy = collection_entropies.chunks_sequence_entropy[collection_average_chunk_id]

            match_lengths_idx = 0
            for token in collection_average_chunk.split[:self.max_tokens_to_display]:
                if token not in collection_average_chunk.words:
                    words_elements.append(self._neutral_style_token(token))
                    continue

                token_repetition = collection_average_sequence_entropy.match_lengths[match_lengths_idx]
                match_lengths_idx += 1
                words_elements.append(self._neutral_style_token(" "))
                words_elements.append(self._style_sequence_token(token, token_repetition))

            words_elements.append(self._neutral_style_token("..."))
            return words_elements
        
        # Visualization 2

        @self.app.callback(
            Output('collection-features-entropy-1', 'children'),
            Input('collection-features-entropy-dropdown-1', 'value'),
        )
        def collection_features_entropy_1(collection_name: str) -> html.Div:
            features = self._find_top_entropy_features(collection_name)
            return _collection_features_entropy(
                collection_name=collection_name,
                features=features
            )

        @self.app.callback(
            Output('collection-features-entropy-2', 'children'),
            Input('collection-features-entropy-dropdown-1', 'value'),
            Input('collection-features-entropy-dropdown-2', 'value'),
        )
        def collection_features_entropy_2(collection_name_1: str, collection_name_2: str) -> html.Div:
            features = self._find_top_entropy_features(collection_name_1)
            return _collection_features_entropy(
                collection_name=collection_name_2, 
                features=features
            )
        
        def _collection_features_entropy(collection_name: str, features: List[str]) -> html.Div:
            words_elements = []
            all_chunks = self.preprocessing_results.get_all_chunks_preprocessing_data()
            collection_entropies = self.entropy_analysis_results.collections_entropies[collection_name]
            collection_average_chunk_id = collection_entropies.average_chunk_id
            collection_average_chunk = [chunk for chunk in all_chunks if collection_average_chunk_id == chunk.chunk_id][0]

            for token in collection_average_chunk.split[:self.max_tokens_to_display]:
                words_elements.append(self._neutral_style_token(" "))
                if token in features:
                    feature_idx = features.index(token)
                    words_elements.append(self._style_feature_token(token, feature_idx))
                else:
                    words_elements.append(self._neutral_style_token(token))

            words_elements.append(self._neutral_style_token("..."))
            return words_elements

    def _find_top_entropy_features(self, collection_name: str) -> List[str]:
        features = self.feature_extractor.get_top_punctuation_features() + self.feature_extractor.get_top_function_words_features()
        collection_entropies = self.entropy_analysis_results.collections_entropies[collection_name]
        collection_average_chunk_id = collection_entropies.average_chunk_id
        collection_features_entropies = collection_entropies.chunks_features_entropies[collection_average_chunk_id].features_entropy
        collection_selected_features_entropies = {key: collection_features_entropies[key] for key in features}
        collection_selected_features_entropies = dict(sorted(collection_selected_features_entropies.items(), key=lambda item: item[1], reverse=True))
        return list(collection_selected_features_entropies.keys())[:self.top_n_features_to_display]
    
    def _style_sequence_token(self, token: str, token_repetition: int) -> html.Span:
        if token_repetition == 1:
            return self._neutral_style_token(token)
        else:
            max_repetition = 10 
            intensity = min(token_repetition / max_repetition, 1.0)
            start_color = (248, 252, 100)  # RGB for #FCF5CF
            end_color = (201, 102, 83)     # RGB for #F2CC00
            red_value = int(start_color[0] + (end_color[0] - start_color[0]) * intensity)
            green_value = int(start_color[1] + (end_color[1] - start_color[1]) * intensity)
            blue_value = int(start_color[2] + (end_color[2] - start_color[2]) * intensity)
            background_color = f'rgb({red_value}, {green_value}, {blue_value})'

        return html.Span(token, style={"background-color": background_color, "color": "black"})
    
    def _style_feature_token(self, token: str, feature_idx: int) -> html.Span:
        # Generate colors based on the number of features to display
        colors = self._generate_colors(self.top_n_features_to_display)

        # Ensure feature_idx is within the range of the colors list
        if 0 <= feature_idx < len(colors):
            color = colors[feature_idx]
            background_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
        else:
            background_color = "#FFFFFF"  # Default to white if index is out of range
        background_color = "#FFFFFF"
        return html.Span(token, style={"background-color": background_color, "color": "black"})

    def _neutral_style_token(self, token: str) -> html.Span:
        return html.Span(token, style={"background-color": "#FFFFFF", "color": "black"})
    
    def _generate_colors(self, n: int) -> list:
        # Generate a list of n distinct colors using a colormap
        cmap = plt.get_cmap('tab10')  # You can choose any colormap
        return [cmap(i / n) for i in range(n)]
    
    def run(self, port: int):
        self.app.run(
            port=port,
            jupyter_height=800,
            debug=False,
        )  


class WritingStyleEntropyAnalysisVisualization(AnalysisVisualization):
    
    def __init__(self, 
                 feature_extractor: FeatureExtractor, 
                 preprocessing_results: WritingStylePreprocessingResults,
                 entropy_analysis_results: EntropyResults
    ) -> None:
        self.entropy_analysis_results = entropy_analysis_results
        self.feature_extractor = feature_extractor
        self.preprocessing_results = preprocessing_results
        self.dash_app = WritingStyleEntropyAnalysisVisualizationDashApp(
            preprocessing_results=preprocessing_results,
            entropy_analysis_results=entropy_analysis_results,
            feature_extractor=feature_extractor
        )

    def visualize(self):
        self._visualize_average_chunks_entropies()
    
    def _visualize_average_chunks_entropies(self):
        data = []

        for collection_name in self.entropy_analysis_results.collection_names:
            collections_entropies = self.entropy_analysis_results.collections_entropies[collection_name]
            average_chunk_id = collections_entropies.average_chunk_id
            collection_chunks_entropies = list(collections_entropies.chunks_features_entropies[average_chunk_id].features_entropy.values())
            collection_chunks_entropies.append(collections_entropies.chunks_sequence_entropy[average_chunk_id].entropy)
            collection_chunks_entropies.append(collections_entropies.chunks_ws_words_entropy[average_chunk_id].entropy)
            collection_chunks_entropies.append(collections_entropies.chunks_all_words_entropy[average_chunk_id].entropy)
            data.append(collection_chunks_entropies)

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=self.feature_extractor.get_feature_names_without_metadata() + ["sequence", "local_word_distribution", "global_word_distribution"],
            y=self.entropy_analysis_results.collection_names,
            colorscale='Viridis',
        ))
        fig.update_layout(title="Heatmap of average chunks' attributes IC values", xaxis_title='Feature', yaxis_title='Collection')
        fig.show()