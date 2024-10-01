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
            dcc.Graph(id='all_chunks', style={'margin-bottom': '0'}),
            html.Div(id='all_chunks_text', style={'background': 'white', "padding": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id='author-dropdown-1',
                    options=[{'label': author, 'value': author} for author in self.pca_analysis_results.author_names],
                    value=self.pca_analysis_results.author_names[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='collection1-dropdown-1',
                    options=[{'label': collection, 'value': collection} for collection in self.pca_analysis_results.collection_names],
                    value=self.pca_analysis_results.collection_names[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='collection2-dropdown-1',
                    options=[{'label': collection, 'value': collection} for collection in self.pca_analysis_results.collection_names],
                    value=self.pca_analysis_results.collection_names[1],
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            dcc.Graph(id='collection_vs_collection_per_author_chunks', style={'margin-bottom': '0'}),
            html.Div(id='collection_vs_collection_per_author_chunks_text', style={'background': 'white', "padding": "10px"}),
            dcc.Dropdown(
                id='author-dropdown-2',
                options=[{'label': author, 'value': author} for author in self.pca_analysis_results.author_names],
                value=self.pca_analysis_results.author_names[0],
                clearable=False,
                style={'width': '100%'}
            ),
            dcc.Graph(id='collections_per_author_chunks'),
            html.Div(id='collections_per_author_chunks_text', style={'background': 'white', "padding": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id='author-dropdown-3',
                    options=[{'label': author, 'value': author} for author in self.pca_analysis_results.author_names],
                    value=self.pca_analysis_results.author_names[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='collection-dropdown-3',
                    options=[{'label': collection, 'value': collection} for collection in self.pca_analysis_results.collection_names],
                    value=self.pca_analysis_results.collection_names[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            dcc.Graph(id='author_collection_chunks'),
            html.Div(id='author_collection_chunks_text', style={'background': 'white', "padding": "10px"}),
        ])

    def setup_callbacks(self):
        @staticmethod
        def _update_annotation_text(top_features: str) -> html.Div:
            return html.Div([
                html.P(f"PC1 Top features: {top_features["PC1"]}", style={'margin': '0', "font-size": "20px"}),
                html.P(f"PC2 Top features: {top_features["PC2"]}", style={'margin': '0', "font-size": "20px"})
            ])
        
        @staticmethod
        def _add_trace(fig: go.Figure, pca: PCAAnalysisResults, collection_name: str) -> None:
            mask = pca.results['collection_name'] == collection_name
            fig.add_trace(go.Scatter(
                x=pca.results.loc[mask, 'PC1'],
                y=pca.results.loc[mask, 'PC2'],
                mode='markers',
                marker=dict(color=PCAAnalysisVisualization.COLLECTION_COLORS[collection_name]),
                name=collection_name,
                text=pca.results.loc[mask, 'author_name'] + " - " + pca.results.loc[mask, 'source_name'],
                hoverinfo='text'
            ))

        @self.app.callback(
            Output('all_chunks', 'figure'),
            Input('author-dropdown-1', 'value'),
        )
        def update_graph_0(_: str):
            fig = go.Figure()
            pca = self.pca_analysis_results.all_chunks

            for collection_name in self.pca_analysis_results.collection_names:
                _add_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'All Chunks PCA Analysis',
                legend_title='Collection',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig

        @self.app.callback(
            Output('all_chunks_text', 'children'),
            Input('author-dropdown-1', 'value'),
        )
        def update_annotation_text_0(_: str):
            top_features = self.pca_analysis_results.all_chunks.top_features
            return _update_annotation_text(top_features)
         
        @self.app.callback(
            Output('collection_vs_collection_per_author_chunks', 'figure'),
            Input('author-dropdown-1', 'value'),
            Input('collection1-dropdown-1', 'value'),
            Input('collection2-dropdown-1', 'value')
        )
        def update_graph_1(selected_author, selected_collection1, selected_collection2):
            fig = go.Figure()
            pca = self.pca_analysis_results.collection_vs_collection_per_author_chunks[selected_author][selected_collection1][selected_collection2]

            for collection_name in [selected_collection1, selected_collection2]:
                _add_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'[{selected_author}] Chunks PCA Analysis',
                legend_title='Collection',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig

        @self.app.callback(
            Output('collection_vs_collection_per_author_chunks_text', 'children'),
            Input('author-dropdown-1', 'value'),
            Input('collection1-dropdown-1', 'value'),
            Input('collection2-dropdown-1', 'value')
        )
        def update_annotation_text_1(selected_author, selected_collection1, selected_collection2):
            top_features = self.pca_analysis_results.collection_vs_collection_per_author_chunks[selected_author][selected_collection1][selected_collection2].top_features
            return _update_annotation_text(top_features)
        
        @self.app.callback(
            Output('collections_per_author_chunks', 'figure'),
            Input('author-dropdown-2', 'value'),
        )
        def update_graph_2(selected_author):
            fig = go.Figure()
            pca = self.pca_analysis_results.collections_per_author_chunks[selected_author]

            for collection_name in self.pca_analysis_results.collection_names:
                _add_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'[{selected_author}] Chunks PCA Analysis',
                legend_title='Collection',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig
        
        @self.app.callback(
            Output('collections_per_author_chunks_text', 'children'),
            Input('author-dropdown-2', 'value'),
        )
        def update_annotation_text_2(selected_author):
            top_features = self.pca_analysis_results.collections_per_author_chunks[selected_author].top_features
            return _update_annotation_text(top_features)
        
        @self.app.callback(
            Output('author_collection_chunks', 'figure'),
            Input('author-dropdown-3', 'value'),
            Input('collection-dropdown-3', 'value'),
        )
        def update_graph_3(selected_author, selected_colleciton):
            fig = go.Figure()
            pca = self.pca_analysis_results.author_collection_chunks[selected_author][selected_colleciton]

            for collection_name in self.pca_analysis_results.collection_names:
                _add_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'[{selected_author}] Chunks PCA Analysis',
                legend_title='Collection',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig

        @self.app.callback(
            Output('author_collection_chunks_text', 'children'),
            Input('author-dropdown-3', 'value'),
            Input('collection-dropdown-3', 'value'),
        )
        def update_annotation_text_3(selected_author, selected_colleciton):
            top_features = self.pca_analysis_results.author_collection_chunks[selected_author][selected_colleciton].top_features
            return _update_annotation_text(top_features)
         
    def run(self):
        self.app.run(
            port=8050,
            jupyter_height=800,
            debug=True,
        )