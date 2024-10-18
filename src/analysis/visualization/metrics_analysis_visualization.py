from dataclasses import fields
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import islice
from src.analysis.metrics.extractor import MetricsExtractor
from src.analysis.metrics.models import MetricData, MetricsAnalysisResults
from src.analysis.visualization.analysis_visualization import AnalysisVisualization
from src.settings import Settings
from dash import dcc, html, Dash
from dash.dependencies import Input, Output

class DashApp:

    class Helper:
        def _get_collections_button(parent: "DashApp", id: str, collection_idx: int = 0) -> dcc.Dropdown:
            return dcc.Dropdown(
                id=id,
                options=[{'label': collection, 'value': collection} for collection in parent.metrics_analysis_results.collection_names],
                value=parent.metrics_analysis_results.collection_names[collection_idx],
                clearable=False,
                style={'width': '100%'}
            )
        
        def _get_authors_button(parent: "DashApp", id: str, author_idx: int = 0) -> dcc.Dropdown:
            return dcc.Dropdown(
                id=id,
                options=[{'label': author, 'value': author} for author in parent.metrics_analysis_results.author_names],
                value=parent.metrics_analysis_results.author_names[author_idx],
                clearable=False,
                style={'width': '100%'}
            )
        
    def __init__(self, settings: Settings, metrics_analysis_results: MetricsAnalysisResults):
        self.metrics_analysis_results = metrics_analysis_results
        self.extractor = MetricsExtractor(settings=settings)
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                DashApp.Helper._get_collections_button(parent=self, id='collection1-dropdown-1'),
                DashApp.Helper._get_authors_button(parent=self, id='authors1-dropdown-1'),
                DashApp.Helper._get_collections_button(parent=self, id='collection2-dropdown-1', collection_idx=1),
                DashApp.Helper._get_authors_button(parent=self, id='authors2-dropdown-1', author_idx=1),
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            dcc.Graph(id='relative_bars', style={'margin-bottom': '0'}),
        ])

    def setup_callbacks(self):

        ### GRAPH 0

        @self.app.callback(
            Output('relative_bars', 'figure'),
            Input('collection1-dropdown-1', 'value'),
            Input('authors1-dropdown-1', 'value'),
            Input('collection2-dropdown-1', 'value'),
            Input('authors2-dropdown-1', 'value'),
        )
        def update_relative_bars(collection_1: str, author_1: str, collection_2: str, author_2: str) -> go.Figure:
            all_full_metrics = self.metrics_analysis_results.get_all_full_metrics()
            top_function_words_names = self.extractor.get_top_function_words_names(all_full_metrics)

            author_1_books_metrics = self.metrics_analysis_results.full_author_collection[author_1][collection_1]
            author_2_books_metrics = self.metrics_analysis_results.full_author_collection[author_2][collection_2]
            author_1_books_features = self.extractor.get_features(
                metrics_data=[author_1_books_metrics],
                top_function_words_names=top_function_words_names
            ).iloc[0]
            author_2_books_features = self.extractor.get_features(
                metrics_data=[author_2_books_metrics],
                top_function_words_names=top_function_words_names
            ).iloc[0]

            feature_names = author_1_books_features.index.to_list()
            for element in ["source_name", "author_name", "collection_name"]:
                feature_names.remove(element)

            relative_bars = {feature_name: self._get_relative_difference(author_1_books_features[feature_name], author_2_books_features[feature_name]) 
                    for feature_name in feature_names}

            colors = [AnalysisVisualization.COLORS["negative"] if value < 0 
                    else AnalysisVisualization.COLORS["positive"] 
                    for value in relative_bars.values()]
            fig = go.Figure(data=[
                go.Bar(
                    name="Mark Twain", 
                    x=list(relative_bars.keys()), 
                    y=list(relative_bars.values()),
                    marker=dict(color=colors),
            )
            ])
            fig.update_layout(
                title="Relative difference of features",
                xaxis_title="Feature",
                yaxis_title="Relative difference",
                font=dict(size=MetricsAnalysisVisualization.FONT_SIZE)
            )

            return fig
        
    def _get_relative_difference(self, value_1, value_2):
        max = np.max([value_1, value_2])
        difference = value_1 - value_2
        relative_difference = difference / max
        return relative_difference
    
    def run(self, port: int):
        self.app.run(
            port=port,
            jupyter_height=800,
            debug=True,
        )

class MetricsAnalysisVisualization(AnalysisVisualization):
    FONT_SIZE = 10

    def __init__(self, settings: Settings, metrics_analysis_results: MetricsAnalysisResults) -> None:
        self.configuration = settings.configuration
        self.metrics_analysis_results = metrics_analysis_results
        self.dash_app = DashApp(settings=settings, metrics_analysis_results=metrics_analysis_results)

    def visualize(self):
        """Visualize the analysis data for the authors and models"""
        self._visualize(self.metrics_analysis_results)
        self._visualize_function_words(self.metrics_analysis_results)
        self._visualize_punctuation_frequency(self.metrics_analysis_results)
        self._visualize_metrics_of_two(self.metrics_analysis_results)

    def _visualize(self, metrics_analysis_results: MetricsAnalysisResults):
        """Visualize the unique_word_counts, average_word_lengths and average_sentence_lengths for the authors and models"""
        fig_xaxes_font_size = 10
        fig_height = 1500
        fig = make_subplots(rows=11, cols=1, subplot_titles=(
                                    "Unique word count", 
                                    "Average word length", 
                                    "Average sentence length", 
                                    "Average syllables per word", 
                                    "Flesch reading ease",
                                    "Flesch Kincaid Grade Level",
                                    "Gunning Fog Index",
                                    "Yules Characteristic K",
                                    "Herdan's C",
                                    "Maas",
                                    "Simpsons_index"
                                ),
                                # vertical_spacing=0.1
                            )

        for i, collection_name in enumerate(metrics_analysis_results.collection_names):
                full_collection_metrics_data = metrics_analysis_results.full_collection_author[collection_name]
                collection_authors_metrics = {
                    "Unique word count": [m.unique_word_count for m in full_collection_metrics_data.values()], 
                    "Average word length": [m.average_word_length for m in full_collection_metrics_data.values()], 
                    "Average sentence length": [m.average_sentence_length for m in full_collection_metrics_data.values()],
                    "Average syllables per word": [m.average_syllables_per_word for m in full_collection_metrics_data.values()], 
                    "Flesch Reading Ease": [m.flesch_reading_ease for m in full_collection_metrics_data.values()], 
                    "Flesch Kincaid Grade Level": [m.flesch_kincaid_grade_level for m in full_collection_metrics_data.values()], 
                    "Gunning Fog Index": [m.gunning_fog_index for m in full_collection_metrics_data.values()],
                    "Yules Characteristic K": [m.yules_characteristic_k for m in full_collection_metrics_data.values()],
                    "Herdan's C": [m.herdans_c for m in full_collection_metrics_data.values()],
                    "Maas": [m.maas for m in full_collection_metrics_data.values()],
                    "Simpsons Index": [m.simpsons_index for m in full_collection_metrics_data.values()]
                }

                for j, (_, value) in enumerate(collection_authors_metrics.items()):
                    fig.add_trace(go.Bar(
                        name=collection_name, 
                        x=metrics_analysis_results.author_names, 
                        marker_color=MetricsAnalysisVisualization.COLLECTION_COLORS_LIST[i],
                        y=value, 
                        showlegend=j==0
                    ), row=j+1, col=1)

        fig.update_xaxes(tickfont_size=fig_xaxes_font_size)
        fig.update_layout(
            legend_title_text="Collection",
            height=fig_height,
        )
        fig.show()

    def _visualize_function_words(self, metrics_analysis_results: MetricsAnalysisResults):
        """Visualize the top 10 function words for the authors and models"""
        fig_font_size = 10
        fig_height = 800
        fig = make_subplots(
            rows=len(metrics_analysis_results.collection_names), cols=10, 
            subplot_titles=metrics_analysis_results.author_names
        )
        max_freq_overall = 0

        for i, collection_name in enumerate(metrics_analysis_results.collection_names):
            full_collection_metrics_data = metrics_analysis_results.full_collection_author[collection_name]
            top_function_words = [dict(islice(d.sorted_function_words.items(), self.configuration.top_n_function_words))
                              for d in full_collection_metrics_data.values()]
            authors_top_function_words = dict(zip(metrics_analysis_results.author_names, top_function_words))

            for j, (_, author_top_function_words) in enumerate(authors_top_function_words.items()):
                max_freq_overall = max([max_freq_overall] + list(author_top_function_words.values()))
                fig.add_trace(go.Bar(
                    name=collection_name, 
                    x=list(author_top_function_words.keys()), 
                    y=list(author_top_function_words.values()), 
                    marker_color=MetricsAnalysisVisualization.COLLECTION_COLORS_LIST[i],
                    showlegend=j==0
                ), row=i+1, col=j+1)

        fig.update_yaxes(range=[0, max_freq_overall])
        fig.update_xaxes(tickfont_size=fig_font_size)
        fig.update_annotations(font_size=fig_font_size)
        fig.update_layout(
            legend_title_text="Collection",
            title_text="Top 10 function words", 
            title_x=0.5,
            height=fig_height
        )
        fig.show()

    def _visualize_punctuation_frequency(self, metrics_analysis_results: MetricsAnalysisResults):
        fig_height = 1500
        total_cols = 3
        total_rows = 4
        fig = make_subplots(
            rows=total_rows, 
            cols=total_cols,
            subplot_titles=metrics_analysis_results.author_names,
        )

        show_legend = True
        for i, collection_name in enumerate(metrics_analysis_results.collection_names):
            full_collection_metrics_data = metrics_analysis_results.full_collection_author[collection_name]
            for j, (author_name, metrics) in enumerate(full_collection_metrics_data.items()):
                row = j // total_cols + 1
                col = j % total_cols + 1
                punctuation_frequency = metrics.punctuation_frequency
                sorted_keys = sorted(punctuation_frequency.keys())
                sorted_values = [punctuation_frequency[key] for key in sorted_keys]
                fig.add_trace(go.Scatter(
                        name=collection_name, 
                        x=list(sorted_keys), 
                        y=list(sorted_values), 
                        marker_color=MetricsAnalysisVisualization.COLLECTION_COLORS_LIST[i],
                        showlegend=show_legend,
                        mode="markers" 
                    ), row=row, col=col
                )
                show_legend = False
            show_legend = True

        fig.update_layout(
            legend_title_text="Collection",
            height=fig_height,
            title_text="Punctuation frequency",
            title_x=0.5
        )     
        fig.show()

    def _visualize_metrics_of_two(self, metrics_analysis_results: MetricsAnalysisResults):
        """Visualize the unique_word_counts, average_word_lengths and average_sentence_lengths for the authors and models"""
        fig = go.Figure()
        excluded_metrics = ["source_name", "author_name", "collection_name", "sorted_function_words", "punctuation_frequency"]
        included_metrics = [f.name for f in fields(MetricData) if f.name not in excluded_metrics]
        buttons = []
        for i, metric_name in enumerate(included_metrics):
            button = dict(
                label=metric_name,
                method='update',
                args=[{'visible': [j == i for j in range(len(included_metrics))]},
                    {'title': metric_name}]
            )
            buttons.append(button)


        for i, collection_name in enumerate(metrics_analysis_results.collection_names):
            full_collection_metrics_data = metrics_analysis_results.full_collection_author[collection_name]
            collection_metrics_per_metric = {metric_name: [] for metric_name in included_metrics}
            for author_name, metrics in full_collection_metrics_data.items():
                for metric_name in collection_metrics_per_metric.keys():
                    collection_metrics_per_metric[metric_name].append(getattr(metrics, metric_name))
            
            for metric_name, value in collection_metrics_per_metric.items():
                fig.add_trace(go.Scatter(
                        name=collection_name, 
                        x=metrics_analysis_results.author_names, 
                        marker_color=MetricsAnalysisVisualization.COLLECTION_COLORS_LIST[i],
                        y=value,
                        mode="markers",
                        visible=(metric_name == included_metrics[0])
                    )
                )

        fig.update_layout(
            legend_title_text="Collection",
            updatemenus=[dict(
                type='dropdown',
                direction='down',
                buttons=buttons,
            )],
            title=included_metrics[0],
        )
        fig.show()
                