from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.analysis_data import AnalysisData
from src.settings import Settings

class AnalysisVisualization():
    LEGEND_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22"]
    COLLECTION_COLORS = {
        "books": "#3498db", 
        "gpt-3.5-turbo-0125": "#e74c3c",
        "gpt-4o": "#2ecc71", 
        "gemini-1.5-flash": "#f1c40f", 
        "open-mixtral-8x7b": "#9b59b6", 
        "claude-3-haiku-20240307": "#e67e22",
    }
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

    def _visualize(self, analysis_data: AnalysisData):
        """Visualize the unique_word_counts, average_word_lengths and average_sentence_lengths for the authors and models"""
        fig_xaxes_font_size = 10
        fig_height = 1000
        fig = make_subplots(rows=7, cols=1, subplot_titles=(
                                    "Unique word count", 
                                    "Average word length", 
                                    "Average sentence length", 
                                    "Average syllables per word", 
                                    "Flesch reading ease",
                                    "Flesch Kincaid Grade Level",
                                    "Gunning Fog Index"
                                ),
                                vertical_spacing=0.1
                            )

        for i, (model_name, metrics) in enumerate(analysis_data.collection_metrics.items()):
            metrics_subset = {
                "Unique word count": [d.unique_word_count for d in metrics], 
                "Average word length": [d.average_word_length for d in metrics], 
                "Average sentence length": [d.average_sentence_length for d in metrics],
                "Average syllables per word": [d.average_syllables_per_word for d in metrics], 
                "Flesch Reading Ease": [d.flesch_reading_ease for d in metrics], 
                "Flesch Kincaid Grade Level": [d.flesch_kincaid_grade_level for d in metrics], 
                "Gunning Fog Index": [d.gunning_fog_index for d in metrics]
            }

            for j, (_, value) in enumerate(metrics_subset.items()):
                fig.add_trace(go.Bar(
                    name=model_name, 
                    x=analysis_data.author_names, 
                    marker_color=AnalysisVisualization.LEGEND_COLORS[i],
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

        for i, (model_name, metrics) in enumerate(analysis_data.collection_metrics.items()):
            author_names = [d.author_name for d in metrics]
            function_words = [self._sort_and_trim_fw_frequency(d.top_10_function_words)
                              for d in metrics]
            authors_function_words = dict(zip(author_names, function_words))

            for j, (_, function_words) in enumerate(authors_function_words.items()):
                max_freq_overall = max([max_freq_overall] + list(function_words.values()))
                fig.add_trace(go.Bar(
                    name=model_name, 
                    x=list(function_words.keys()), 
                    y=list(function_words.values()), 
                    marker_color=AnalysisVisualization.LEGEND_COLORS[i],
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
        for i, (model_name, metrics) in enumerate(analysis_data.collection_metrics.items()):
            for j, author_metrics in enumerate(metrics):
                row = j // total_cols + 1
                col = j % total_cols + 1
                punctuation_frequency = author_metrics.punctuation_frequency
                sorted_keys = sorted(punctuation_frequency.keys())
                sorted_values = [punctuation_frequency[key] for key in sorted_keys]
                fig.add_trace(go.Scatter(
                        name=model_name, 
                        x=list(sorted_keys), 
                        y=list(sorted_values), 
                        marker_color=AnalysisVisualization.LEGEND_COLORS[i],
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
        # Create the scatter plot
        fig = go.Figure()
        df = analysis_data.pca_results
        pc1_column = next(column for column in df.columns if column.startswith('PC1'))
        pc2_column = next(column for column in df.columns if column.startswith('PC2'))

        for collection_name in analysis_data.collection_names:
            mask = df['collection_name'] == collection_name
            fig.add_trace(go.Scatter(
                x=df.loc[mask, pc1_column],
                y=df.loc[mask, pc2_column],
                mode='markers',
                marker=dict(color=AnalysisVisualization.COLLECTION_COLORS[collection_name]),
                name=collection_name,
                text=df.loc[mask, 'author_name'], 
                hoverinfo='text'
            ))

        # Update layout with buttons
        fig.update_layout(
            title='PCA Analysis',
            xaxis_title='PC1 [25.32%]',
            yaxis_title='PC2 [10.03%]',
            legend_title='Collection Name',
        )

        # Show the plot
        fig.show()

    def _visualize_pca_by_authors(self, analysis_data: AnalysisData):
        """Visualize the PCA data for the authors and models"""
        # Create the scatter plot
        fig = go.Figure()
        df = analysis_data.pca_results
        pc1_column = next(column for column in df.columns if column.startswith('PC1'))
        pc2_column = next(column for column in df.columns if column.startswith('PC2'))

        for author_name in analysis_data.author_names:
            mask = df['author_name'] == author_name
            fig.add_trace(go.Scatter(
                x=df.loc[mask, pc1_column],
                y=df.loc[mask, pc2_column],
                mode='markers',
                marker=dict(color=AnalysisVisualization.AUTHOR_COLORS[author_name]),
                name=author_name,
                text=df.loc[mask, 'collection_name'], 
                hoverinfo='text'
            ))

        # Update layout with buttons
        fig.update_layout(
            title='PCA Analysis',
            xaxis_title='PC1 [25.32%]',
            yaxis_title='PC2 [10.03%]',
            legend_title='Author Name',
        )

        # Show the plot
        fig.show()

    def _sort_and_trim_fw_frequency(self, fw_frequency: Dict[str, int]) -> Dict[str, int]:
        """Sort the function words frequency"""
        sorted_fw_frequency = sorted(fw_frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_fw_frequency[:self.configuration.n_top_function_words])
    
    def _visualize_function_words_heatmap(self, data: Dict[str, List[AnalysisData]]):
        _, first_data = next(iter(data.items()))
        model_names = list(data.keys())
        params = {}
        for y_idx in range(1, 10*3+1, 10):
            params[f"yaxis{y_idx}"] = go.YAxis(
                title=model_names[3 + y_idx//10], 
                titlefont=go.Font(size=25)
            )
        layout = go.Layout(**params)
        fig = make_subplots(
            rows=3, 
            cols=10, 
            horizontal_spacing=0.075,
            vertical_spacing=0.05,
            subplot_titles=[data.author_name for data in first_data],
            figure=go.Figure(layout=layout),
        )
        init_legend = True

        for i, (model_name, analysis_data) in enumerate(data.items()):
            author_names = [d.author_name for d in analysis_data]
            function_words = [d.top_10_function_words for d in analysis_data]
            authors_function_words = dict(zip(author_names, function_words))
            if i < 3:
                continue

            for j, (_, function_words) in enumerate(authors_function_words.items()):
                values = list(reversed(list(function_words.values())))
                values_matrix = np.array(values[:10]).reshape(10, 1)    
                fig.add_trace(go.Heatmap(
                    z=values_matrix,
                    y=list(reversed(list(function_words.keys()))),
                    zmin=0,
                    zmax=1000,
                    showscale=init_legend, 
                ), row=i+1-3, col=j+1)

                init_legend = False

        # fig.update_traces(showlegend=True, showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(tickfont_size=20)
        fig.update_traces(colorbar_orientation="h",
                            selector=dict(type="heatmap"),
                            colorscale="oranges",
                            colorbar=dict(
                                x=0.5, 
                                y=1.3,
                                tickfont=dict(size=16)  
                            )
                        )
        fig.update_annotations(font=dict(size=20),  # Optional: Adjust font size as needed
                            textangle=65)  #
        fig.update_layout(
            height=1500,
        )
        fig.show()

    def _visualize_function_words_large(self, data: Dict[str, List[AnalysisData]]):
        """Visualize the top 10 function words for the authors and models"""
        _, first_data = next(iter(data.items()))
        author_names = [d.author_name for d in first_data]
        params = {}
        for y_idx in range(1, 10*6+1, 6):
            params[f"yaxis{y_idx}"] = go.YAxis(
                title=author_names[y_idx//6], 
                titlefont=go.Font(size=AnalysisVisualization.FONT_SIZE_EXTRA_LARGE)
            )
        layout = go.Layout(**params)
        fig = make_subplots(
            rows=5, 
            cols=6, 
            subplot_titles=[model_name for model_name in data.keys()],
            figure=go.Figure(layout=layout),
            vertical_spacing=0.02 
        )
        max_freq_overall = 0

        for i, (model_name, analysis_data) in enumerate(data.items()):
            author_names = [d.author_name for d in analysis_data]
            function_words = [d.top_10_function_words for d in analysis_data]
            authors_function_words = dict(zip(author_names, function_words))

            for j, (_, function_words) in enumerate(authors_function_words.items()):
                max_freq_overall = max([max_freq_overall] + list(function_words.values()))
                fig.add_trace(go.Bar(
                    name=model_name, 
                    y=list(reversed(list(function_words.keys()))), 
                    x=list(reversed(list(function_words.values()))), 
                    marker_color=AnalysisVisualization.LEGEND_COLORS[i],
                    orientation="h",
                    showlegend=False
                ), row=j+1, col=i+1)

        fig.update_xaxes(
            range=[0, max_freq_overall],
            tickfont_size=AnalysisVisualization.FONT_SIZE_MEDIUM
        )
        fig.update_yaxes(tickfont_size=AnalysisVisualization.FONT_SIZE_EXTRA_LARGE)
        fig.update_annotations(font_size=AnalysisVisualization.FONT_SIZE_LARGE)
        fig.update_layout(
            title_text="Top 10 function words", 
            title_x=0.5,
            title_font_size=AnalysisVisualization.FONT_SIZE_TITLE,
            height=3000,
        )
        fig.show()