from enum import Enum
import plotly.graph_objects as go
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from src.analysis.visualization.metrics_analysis_visualization import AnalysisVisualization
from src.analysis.pca.data import PCAAnalysisResults

class DashApp:
    
    class Helper:
        @staticmethod
        def _get_mark_by_button(id: str) -> dcc.Dropdown:
            return dcc.Dropdown(
                id=id,
                options=[
                    {'label': 'Mark by authors', 'value': 'AUTHORS'},
                    {'label': 'Mark by collections', 'value': 'COLLECTIONS'}
                ],
                value='COLLECTIONS',
                clearable=False,
                style={'width': '100%'}
            )

        def _get_collections_button(parent: "PCAAnalysisVisualization", id: str, collection_idx: int = 0) -> dcc.Dropdown:
            return dcc.Dropdown(
                id=id,
                options=[{'label': collection, 'value': collection} for collection in parent.pca_analysis_results.collection_names],
                value=parent.pca_analysis_results.collection_names[collection_idx],
                clearable=False,
                style={'width': '100%'}
            )
        
        def _get_authors_button(parent: "PCAAnalysisVisualization", id: str) -> dcc.Dropdown:
            return dcc.Dropdown(
                id=id,
                options=[{'label': author, 'value': author} for author in parent.pca_analysis_results.author_names],
                value=parent.pca_analysis_results.author_names[0],
                clearable=False,
                style={'width': '100%'}
            )

        @staticmethod
        def _update_annotation_text(top_features: str) -> html.Div:
            return html.Div([
                html.P(f"PC1 Top features: {top_features["PC1"].index.tolist()}", style={'margin': '0', "font-size": "20px"}),
                html.P(f"PC2 Top features: {top_features["PC2"].index.tolist()}", style={'margin': '0', "font-size": "20px"})
            ])
        
        @staticmethod
        def _add_collection_trace(fig: go.Figure, pca: PCAAnalysisResults, collection_name: str) -> None:
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

        @staticmethod
        def _add_author_trace(fig: go.Figure, pca: PCAAnalysisResults, author_name: str) -> None:
            mask = pca.results['author_name'] == author_name
            fig.add_trace(go.Scatter(
                x=pca.results.loc[mask, 'PC1'],
                y=pca.results.loc[mask, 'PC2'],
                mode='markers',
                marker=dict(color=PCAAnalysisVisualization.AUTHOR_COLORS[author_name]),
                name=author_name,
                text=pca.results.loc[mask, 'collection_name'] + " - " + pca.results.loc[mask, 'source_name'],
                hoverinfo='text'
            ))


    def __init__(self, pca_analysis_results: PCAAnalysisResults):
        self.pca_analysis_results = pca_analysis_results
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            # GRAPH 0
            DashApp.Helper._get_mark_by_button('mark-by-0'),
            dcc.Graph(id='all_chunks', style={'margin-bottom': '0'}),
            html.Div(id='all_chunks_text', style={'background': 'white', "padding": "10px"}),
            # GRAPH 1
            DashApp.Helper._get_authors_button(parent=self, id='author-dropdown-1'),
            dcc.Graph(id='authors_chunks'),
            html.Div(id='authors_chunks_text', style={'background': 'white', "padding": "10px"}),
            # GRAPH 2
            DashApp.Helper._get_collections_button(parent=self, id='collection-dropdown-2'),
            dcc.Graph(id='collections_chunks'),
            html.Div(id='collections_chunks_text', style={'background': 'white', "padding": "10px"}),
            # GRAPH 3
            html.Div([
                DashApp.Helper._get_authors_button(parent=self, id='author-dropdown-3'),
                DashApp.Helper._get_collections_button(parent=self, id='collection1-dropdown-3'),
                DashApp.Helper._get_collections_button(parent=self, id='collection2-dropdown-3', collection_idx=1),
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            dcc.Graph(id='author_collection_collection_chunks', style={'margin-bottom': '0'}),
            html.Div(id='author_collection_collection_chunks_text', style={'background': 'white', "padding": "10px"}),
        ])

    def setup_callbacks(self):

        ### GRAPH 0

        @self.app.callback(
            Output('all_chunks', 'figure'),
            Input('mark-by-0', 'value'),
        )
        def update_graph_0(marked_by: str):
            fig = go.Figure()
            pca = self.pca_analysis_results.all_chunks

            if marked_by == "AUTHORS":
                print("I am hereee")
                for author_name in self.pca_analysis_results.author_names:
                    DashApp.Helper._add_author_trace(fig, pca, author_name)
            elif marked_by == "COLLECTIONS":
                print("I am hereee")
                for collection_name in self.pca_analysis_results.collection_names:
                    DashApp.Helper._add_collection_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'All Chunks PCA Analysis {marked_by}' ,
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
            return DashApp.Helper._update_annotation_text(top_features)

        ### GRAPH 1
  
        @self.app.callback(
            Output('authors_chunks', 'figure'),
            Input('author-dropdown-1', 'value'),
        )
        def update_graph_1(selected_author):
            fig = go.Figure()
            pca = self.pca_analysis_results.authors_chunks[selected_author]

            for collection_name in self.pca_analysis_results.collection_names:
                DashApp.Helper._add_collection_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'[{selected_author}] Chunks PCA Analysis',
                legend_title='Collection',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig
        
        @self.app.callback(
            Output('authors_chunks_text', 'children'),
            Input('author-dropdown-1', 'value'),
        )
        def update_annotation_text_1(selected_author):
            top_features = self.pca_analysis_results.authors_chunks[selected_author].top_features
            return DashApp.Helper._update_annotation_text(top_features)
        
        ### GRAPH 2

        @self.app.callback(
            Output('collections_chunks', 'figure'),
            Input('collection-dropdown-2', 'value'),
        )
        def update_graph_2(selected_collection: str) -> go.Figure:
            fig = go.Figure()
            pca = self.pca_analysis_results.collections_chunks[selected_collection]

            for author_name in self.pca_analysis_results.author_names:
                DashApp.Helper._add_author_trace(fig, pca, author_name)

            fig.update_layout(
                title=f'[{selected_collection}] Chunks PCA Analysis',
                legend_title='Authors',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig
        
        @self.app.callback(
            Output('collections_chunks_text', 'children'),
            Input('collection-dropdown-2', 'value'),
        )
        def update_annotation_text_1(selected_collection: str):
            top_features = self.pca_analysis_results.collections_chunks[selected_collection].top_features
            return DashApp.Helper._update_annotation_text(top_features)
        
        ### GRAPH 3

        @self.app.callback(
            Output('author_collection_collection_chunks', 'figure'),
            Input('author-dropdown-3', 'value'),
            Input('collection1-dropdown-3', 'value'),
            Input('collection2-dropdown-3', 'value')
        )
        def update_graph_3(selected_author, selected_collection1, selected_collection2):
            fig = go.Figure()
            pca = self.pca_analysis_results.author_collection_collection_chunks[selected_author][selected_collection1][selected_collection2]

            for collection_name in [selected_collection1, selected_collection2]:
                DashApp.Helper._add_collection_trace(fig, pca, collection_name)

            fig.update_layout(
                title=f'[{selected_author}][{selected_collection1}][{selected_collection2}] Chunks PCA Analysis',
                legend_title='Collection',
                xaxis_title=f'PC1[{pca.pc_variance[0]:.2f}]',
                yaxis_title=f'PC2[{pca.pc_variance[1]:.2f}]',
            )
            return fig

        @self.app.callback(
            Output('author_collection_collection_chunks_text', 'children'),
            Input('author-dropdown-3', 'value'),
            Input('collection1-dropdown-3', 'value'),
            Input('collection2-dropdown-3', 'value')
        )
        def update_annotation_text_3(selected_author, selected_collection1, selected_collection2):
            top_features = self.pca_analysis_results.author_collection_collection_chunks[selected_author][selected_collection1][selected_collection2].top_features
            return DashApp.Helper._update_annotation_text(top_features)
        
    def run(self, port: int):
        self.app.run(
            port=port,
            jupyter_height=800,
            debug=True,
        )

class PCAAnalysisVisualization(AnalysisVisualization):
    
    def __init__(self, pca_analysis_results: PCAAnalysisResults):
        self.dash_app = DashApp(pca_analysis_results)
    
    def visualize_top_features(pca_analysis_results: PCAAnalysisResults) -> None:
        top_features = pca_analysis_results.all_chunks.top_features

        # Create subplots
        fig = make_subplots(
            rows=1, 
            cols=len(top_features), 
            horizontal_spacing=0.25,
            subplot_titles=[
                f'Top Features for {pc}[{pca_analysis_results.all_chunks.pc_variance[i]:.2f}]' 
                for i, pc in enumerate(top_features.keys())
            ]
        )

        for i, (pc, features) in enumerate(top_features.items(), start=1):
            # Extract features and their corresponding importance
            feature_names = list(features.keys())
            importances = list(features.values)

            # Add a vertical bar plot to the subplot
            fig.add_trace(go.Bar(
                x=importances,
                y=feature_names,
                orientation='h',
                marker=dict(color='skyblue'),
                name=f'Top Features for {pc}[{pca_analysis_results.all_chunks.pc_variance[i-1]:.2f}]'
            ), row=1, col=i)

            fig.update_xaxes(range=[min(importances), max(importances)], row=1, col=i)

        # Customize the layout
        fig.update_layout(
            title='Top Features for Principal Components',
            xaxis_title='Importance',
            yaxis_title='Features',
            showlegend=False
        )

        # Invert y-axis for all subplots to have the most important feature on top
        for i in range(1, len(top_features) + 1):
            fig.update_yaxes(autorange='reversed', row=1, col=i)

        # Show the plot
        fig.show()