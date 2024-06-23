from typing import Dict, List
from src.analysis.alanysis_data import AnalysisData
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AnalysisVisualization():
    LEGEND_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22"]
    LEGEND_TITLE = "Text source"
    SUBPLOT_TITLES = ("Word counts", "Average word length", "Unique word count", "Average sentence length")
    FONT_SIZE = 10

    def visualize(self, data: Dict[str, List[AnalysisData]]):
        """Visualize the analysis data for the authors and models"""
        self._visualize(data)
        self._visualize_function_words(data)

    def _visualize(self, data: Dict[str, List[AnalysisData]]):
        """Visualize the word_counts, unique_word_counts, average_word_lengths and average_sentence_lengths for the authors and models"""
        fig = make_subplots(rows=2, cols=2, subplot_titles=AnalysisVisualization.SUBPLOT_TITLES, vertical_spacing=0.4)

        for i, (model_name, analysis_data) in enumerate(data.items()):
            author_names = [d.author_name for d in analysis_data]
            word_counts = [d.word_count for d in analysis_data]
            unique_word_counts = [d.unique_word_count for d in analysis_data]
            average_word_lengths = [d.average_word_length for d in analysis_data]
            average_sentence_lengths = [d.average_sentence_length for d in analysis_data]

            if model_name != "books":
                fig.add_trace(go.Bar(
                    name=model_name, 
                    x=author_names, 
                    y=word_counts, 
                    marker_color=AnalysisVisualization.LEGEND_COLORS[i],
                    showlegend=False
                ), row=1, col=1)
            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=unique_word_counts, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i],
            ), row=2, col=1)
            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=average_word_lengths, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i], 
                showlegend=False
            ), row=1, col=2)
            fig.add_trace(go.Bar(
                name=model_name, 
                x=author_names, 
                y=average_sentence_lengths, 
                marker_color=AnalysisVisualization.LEGEND_COLORS[i], 
                showlegend=False
            ), row=2, col=2)

        fig.update_xaxes(tickfont_size=AnalysisVisualization.FONT_SIZE)
        fig.update_layout(legend_title_text=AnalysisVisualization.LEGEND_TITLE)
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
            function_words = [d.top_10_function_words for d in analysis_data]
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