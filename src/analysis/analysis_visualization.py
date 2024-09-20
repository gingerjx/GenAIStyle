from dataclasses import fields
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import islice
from src.analysis.analysis_data import AnalysisData, MetricData
from src.settings import Settings

class AnalysisVisualization():
    COLLECTION_COLORS = {
        "books": "#3498db", 
        "gpt-3.5-turbo-0125": "#e74c3c",
        "gpt-4o": "#2ecc71", 
        "gemini-1.5-flash": "#f1c40f", 
        "open-mixtral-8x7b": "#9b59b6", 
        "claude-3-haiku-20240307": "#e67e22",
    }
    COLLECTION_COLORS_LIST = list(COLLECTION_COLORS.values())
    AUTHOR_COLORS = {
        "Mark Twain": "#3498db",          # Blue
        "Zane Grey": "#e74c3c",           # Red
        "Joseph Conrad": "#2ecc71",       # Green
        "George Eliot": "#f1c40f",        # Yellow
        "Benjamin Disraeli": "#9b59b6",   # Purple
        "Lucy Maud Montgomery": "#e67e22",# Orange
        "William Henry Hudson": "#1abc9c",# Turquoise
        "Howard Pyle": "#34495e",         # Dark Blue
        "Virginia Woolf": "#d35400",      # Dark Orange
        "Lewis Carroll": "#7f8c8d"        # Gray
    }
    LEGEND_TITLE = "Text source"
    FONT_SIZE = 10

    def __init__(self, settings: Settings) -> None:
        self.configuration = settings.configuration

    def visualize(self, analysis_data: AnalysisData):
        """Visualize the analysis data for the authors and models"""
        self._visualize(analysis_data)
        self._visualize_function_words(analysis_data)
        self._visualize_punctuation_frequency(analysis_data)
        self._visualize_pca_by_collections(analysis_data)
        self._visualize_pca_by_authors(analysis_data)
        self._visualize_metrics_of_two(analysis_data)

    def _visualize(self, analysis_data: AnalysisData):
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

        for i, (collection_name, authors_metrics) in enumerate(analysis_data.collection_author_metrics.items()):
            metrics_subset = {
                "Unique word count": [m.unique_word_count for m in authors_metrics.values()], 
                "Average word length": [m.average_word_length for m in authors_metrics.values()], 
                "Average sentence length": [m.average_sentence_length for m in authors_metrics.values()],
                "Average syllables per word": [m.average_syllables_per_word for m in authors_metrics.values()], 
                "Flesch Reading Ease": [m.flesch_reading_ease for m in authors_metrics.values()], 
                "Flesch Kincaid Grade Level": [m.flesch_kincaid_grade_level for m in authors_metrics.values()], 
                "Gunning Fog Index": [m.gunning_fog_index for m in authors_metrics.values()],
                "Yules Characteristic K": [m.yules_characteristic_k for m in authors_metrics.values()],
                "Herdan's C": [m.herdans_c for m in authors_metrics.values()],
                "Maas": [m.maas for m in authors_metrics.values()],
                "Simpsons Index": [m.simpsons_index for m in authors_metrics.values()]
            }

            for j, (_, value) in enumerate(metrics_subset.items()):
                fig.add_trace(go.Bar(
                    name=collection_name, 
                    x=analysis_data.author_names, 
                    marker_color=AnalysisVisualization.COLLECTION_COLORS_LIST[i],
                    y=value, 
                    showlegend=j==0
                ), row=j+1, col=1)

        fig.update_xaxes(tickfont_size=fig_xaxes_font_size)
        fig.update_layout(
            legend_title_text=AnalysisVisualization.LEGEND_TITLE,
            height=fig_height,
        )
        fig.show()

    def _visualize_function_words(self, analysis_data: AnalysisData):
        """Visualize the top 10 function words for the authors and models"""
        fig_font_size = 10
        fig_height = 800
        fig = make_subplots(
            rows=6, cols=10, 
            subplot_titles=analysis_data.author_names
        )
        max_freq_overall = 0

        for i, (collection_name, authors_metrics) in enumerate(analysis_data.collection_author_metrics.items()):
            author_names = [d.author_name for d in authors_metrics.values()]
            top_function_words = [dict(islice(d.sorted_function_words.items(), self.configuration.top_n_function_words))
                              for d in authors_metrics.values()]
            authors_top_function_words = dict(zip(author_names, top_function_words))

            for j, (_, author_top_function_words) in enumerate(authors_top_function_words.items()):
                max_freq_overall = max([max_freq_overall] + list(author_top_function_words.values()))
                fig.add_trace(go.Bar(
                    name=collection_name, 
                    x=list(author_top_function_words.keys()), 
                    y=list(author_top_function_words.values()), 
                    marker_color=AnalysisVisualization.COLLECTION_COLORS_LIST[i],
                    showlegend=j==0
                ), row=i+1, col=j+1)

        fig.update_yaxes(range=[0, max_freq_overall])
        fig.update_xaxes(tickfont_size=fig_font_size)
        fig.update_annotations(font_size=fig_font_size)
        fig.update_layout(
            title_text="Top 10 function words", 
            title_x=0.5,
            height=fig_height
        )
        fig.show()

    def _visualize_punctuation_frequency(self, analysis_data: AnalysisData):
        fig_height = 1500
        total_cols = 3
        total_rows = 4
        fig = make_subplots(
            rows=total_rows, 
            cols=total_cols,
            subplot_titles=analysis_data.author_names,
        )

        show_legend = True
        for i, (collection_name, authors_metrics) in enumerate(analysis_data.collection_author_metrics.items()):
            for j, (author_name, metrics) in enumerate(authors_metrics.items()):
                row = j // total_cols + 1
                col = j % total_cols + 1
                punctuation_frequency = metrics.punctuation_frequency
                sorted_keys = sorted(punctuation_frequency.keys())
                sorted_values = [punctuation_frequency[key] for key in sorted_keys]
                fig.add_trace(go.Scatter(
                        name=collection_name, 
                        x=list(sorted_keys), 
                        y=list(sorted_values), 
                        marker_color=AnalysisVisualization.COLLECTION_COLORS_LIST[i],
                        showlegend=show_legend,
                        mode="markers" 
                    ), row=row, col=col
                )
                show_legend = False
            show_legend = True

        fig.update_layout(
            height=fig_height,
            title_text="Punctuation frequency",
            title_x=0.5
        )     
        fig.show()

    def _visualize_pca_by_collections(self, analysis_data: AnalysisData):
        """Visualize the PCA data for the authors and models"""
        fig = go.Figure()
        df = analysis_data.pca.results

        for collection_name in analysis_data.collection_names:
            mask = df['collection_name'] == collection_name
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'PC1'],
                y=df.loc[mask, 'PC2'],
                mode='markers',
                marker=dict(color=AnalysisVisualization.COLLECTION_COLORS[collection_name]),
                name=collection_name,
                text=df.loc[mask, 'author_name'], 
                hoverinfo='text'
            ))

        fig.update_layout(
            title='PCA Analysis',
            xaxis_title=f'PC1 [{analysis_data.pca.pc_variance[0]:.2%}]',
            yaxis_title=f'PC2 [{analysis_data.pca.pc_variance[1]:.2%}]',
            legend_title='Collection Name',
        )
        fig.show()

    def _visualize_pca_by_authors(self, analysis_data: AnalysisData):
        """Visualize the PCA data for the authors and models"""
        fig = go.Figure()
        df = analysis_data.pca.results

        for author_name in analysis_data.author_names:
            mask = df['author_name'] == author_name
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'PC1'],
                y=df.loc[mask, 'PC2'],
                mode='markers',
                marker=dict(color=AnalysisVisualization.AUTHOR_COLORS[author_name]),
                name=author_name,
                text=df.loc[mask, 'collection_name'], 
                hoverinfo='text'
            ))

        fig.update_layout(
            title='PCA Analysis',
            xaxis_title=f'PC1 [{analysis_data.pca.pc_variance[0]:.2%}]',
            yaxis_title=f'PC2 [{analysis_data.pca.pc_variance[1]:.2%}]',
            legend_title='Author Name',
        )
        fig.show()

    def _visualize_metrics_of_two(self, analysis_data: AnalysisData):
        """Visualize the unique_word_counts, average_word_lengths and average_sentence_lengths for the authors and models"""
        fig = go.Figure()
        excluded_metrics = ["author_name", "collection_name", "sorted_function_words", "punctuation_frequency"]
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

        for i, (collection_name, authors_metrics) in enumerate(analysis_data.collection_author_metrics.items()):
        
            collection_metrics_per_metric = {metric_name: [] for metric_name in included_metrics}
            for author_name, metrics in authors_metrics.items():
                for metric_name in collection_metrics_per_metric.keys():
                    collection_metrics_per_metric[metric_name].append(getattr(metrics, metric_name))
            
            for metric_name, value in collection_metrics_per_metric.items():
                fig.add_trace(go.Scatter(
                        name=collection_name, 
                        x=analysis_data.author_names, 
                        marker_color=AnalysisVisualization.COLLECTION_COLORS_LIST[i],
                        y=value,
                        mode="markers",
                        visible=(metric_name == included_metrics[0])
                    )
                )

        fig.update_layout(
            updatemenus=[dict(
                type='dropdown',
                direction='down',
                buttons=buttons,
            )],
            title=included_metrics[0],
        )
        fig.show()