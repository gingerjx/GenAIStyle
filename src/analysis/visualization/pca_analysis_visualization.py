import plotly.graph_objects as go
from dash import dcc, html, Dash
from dash.dependencies import Input, Output

from src.analysis.visualization.metrics_analysis_visualization import AnalysisVisualization
from src.analysis.pca.data import PCAAnalysisResults

class PCAAnalysisVisualization(AnalysisVisualization):

    def __init__(self, pca_analysis_results: PCAAnalysisResults):
        self.pca_analysis_results = pca_analysis_results
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                dcc.Dropdown(
                    id='author-dropdown',
                    options=[{'label': author, 'value': author} for author in self.pca_analysis_results.author_names],
                    value=self.pca_analysis_results.author_names[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='collection1-dropdown',
                    options=[{'label': collection, 'value': collection} for collection in self.pca_analysis_results.collection_names],
                    value=self.pca_analysis_results.collection_names[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='collection2-dropdown',
                    options=[{'label': collection, 'value': collection} for collection in self.pca_analysis_results.collection_names],
                    value=self.pca_analysis_results.collection_names[1],
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            dcc.Graph(id='pca-graph')
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('pca-graph', 'figure'),
            Input('author-dropdown', 'value'),
            Input('collection1-dropdown', 'value'),
            Input('collection2-dropdown', 'value')
        )
        def update_graph(selected_author, selected_collection1, selected_collection2):
            fig = go.Figure()
            results = self.pca_analysis_results.collection_vs_collection_per_author_chunks[selected_author][selected_collection1][selected_collection2].results

            for collection_name in [selected_collection1, selected_collection2]:
                mask = results['collection_name'] == collection_name
                fig.add_trace(go.Scatter(
                    x=results.loc[mask, 'PC1'],
                    y=results.loc[mask, 'PC2'],
                    mode='markers',
                    marker=dict(color=PCAAnalysisVisualization.COLLECTION_COLORS[collection_name]),
                    name=collection_name,
                    text=results.loc[mask, 'source_name'],
                    hoverinfo='text'
                ))

            fig.update_layout(
                title=f'[{selected_author}] Chunks PCA Analysis',
                legend_title='Collection',
            )
            return fig

    def run(self):
        self.app.run_server(debug=True)