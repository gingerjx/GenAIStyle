from typing import Dict, List

import numpy as np
from src.analysis.alanysis_data import AnalysisData
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.settings import Settings

class AnalysisVisualization():
    LEGEND_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22"]
    LEGEND_TITLE = "Text source"
    SUBPLOT_TITLES = ("Average word length", "Unique word count", "Average sentence length", "Average syllables per word")
    FONT_SIZE = 10

    def __init__(self, settings: Settings) -> None:
        self.configuration = settings.configuration

    def visualize(self, data: Dict[str, List[AnalysisData]]):
        """Visualize the analysis data for the authors and models"""
        self._visualize(data)
        self._visualize_function_words(data)
        self._visualize_punctuation_frequency(data)

    def _visualize(self, data: Dict[str, List[AnalysisData]]):
        """Visualize the unique_word_counts, average_word_lengths and average_sentence_lengths for the authors and models"""
        fig = make_subplots(rows=4, cols=1, subplot_titles=AnalysisVisualization.SUBPLOT_TITLES)

        for i, (model_name, analysis_data) in enumerate(data.items()):
            author_names = [d.author_name for d in analysis_data]
            unique_word_counts = [d.unique_word_count for d in analysis_data]
            average_word_lengths = [d.average_word_length for d in analysis_data]
            average_sentence_lengths = [d.average_sentence_length for d in analysis_data]
            average_syllables_per_words = [d.average_syllables_per_word for d in analysis_data]

            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=unique_word_counts, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i],
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=average_word_lengths, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i], 
                showlegend=False
            ), row=2, col=1)
            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=average_sentence_lengths, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i], 
                showlegend=False
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=average_syllables_per_words, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i], 
                showlegend=False
            ), row=4, col=1)

        fig.update_xaxes(tickfont_size=AnalysisVisualization.FONT_SIZE)
        fig.update_layout(
            legend_title_text=AnalysisVisualization.LEGEND_TITLE,
            height=800
        )
        fig.show()

    def _visualize_function_words(self, data: Dict[str, List[AnalysisData]]):
        """Visualize the top 10 function words for the authors and models"""
        _, first_data = next(iter(data.items()))
        fig = make_subplots(
            rows=6, 
            cols=10, 
            subplot_titles=[data.author_name for data in first_data]
        )
        max_freq_overall = 0

        for i, (model_name, analysis_data) in enumerate(data.items()):
            author_names = [d.author_name for d in analysis_data]
            function_words = [self._sort_and_trim_fw_frequency(d.top_10_function_words)
                              for d in analysis_data]
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
        fig.update_xaxes(tickfont_size=AnalysisVisualization.FONT_SIZE)
        fig.update_annotations(font_size=AnalysisVisualization.FONT_SIZE)
        fig.update_layout(title_text="Top 10 function words", title_x=0.5)
        fig['layout'].update(height=800)
        fig.show()

    def _visualize_punctuation_frequency(self, data: Dict[str, List[AnalysisData]]):
        total_rows = 4
        total_cols = 3
        _, first_data = next(iter(data.items()))
        show_legend = True
        fig = make_subplots(
            rows=total_rows, 
            cols=total_cols,
            subplot_titles=[data.author_name for data in first_data],
        )

        for i, (model_name, analysis_data) in enumerate(data.items()):
            for j, analysis in enumerate(analysis_data):
                row = j // total_cols + 1
                col = j % total_cols + 1
                punctuation_frequency = analysis.punctuation_frequency
                sorted_keys = sorted(punctuation_frequency.keys())
                sorted_values = [punctuation_frequency[key] for key in sorted_keys]
                fig.add_trace(go.Scatter(
                        name=model_name, 
                        x=list(sorted_keys), 
                        y=list(sorted_values), 
                        marker_color=AnalysisVisualization.LEGEND_COLORS[i],
                        showlegend=show_legend,
                        mode='markers' 
                    ), row=row, col=col
                )
                show_legend = False
            show_legend = True

        fig.update_layout(
            height=1500,
            title_text="Punctuation frequency",
            title_x=0.5
        )     
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
            params[f'yaxis{y_idx}'] = go.YAxis(
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
        fig.update_traces(colorbar_orientation='h',
                            selector=dict(type='heatmap'),
                            colorscale='oranges',
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
            params[f'yaxis{y_idx}'] = go.YAxis(
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
                    orientation='h',
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