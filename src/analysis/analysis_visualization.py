from typing import List
from src.analysis.alanysis_data import AnalysisData
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AnalysisVisualization():
    
    def visualize(self, data: List[AnalysisData], title = None):
        # https://stackoverflow.com/questions/29046057/plotly-grouped-bar-chart-with-multiple-axes
        c_names = [d.collection_name for d in data]
        word_counts = [d.word_count for d in data]
        unique_word_counts = [d.unique_word_count for d in data]
        average_word_lengths = [d.average_word_length for d in data]
        average_sentence_lengths = [d.average_sentence_length for d in data]

        fig = make_subplots(rows=2, cols=2, subplot_titles=("Word counts", "Unique word count", "Average word length", "Average sentence length"))

        fig.add_trace(go.Bar(name='Word counts', x=c_names, y=word_counts), row=1, col=1)
        fig.add_trace(go.Bar(name='Unique word count', x=c_names, y=unique_word_counts), row=1, col=2)
        fig.add_trace(go.Bar(name='Average word length', x=c_names, y=average_word_lengths), row=2, col=1)
        fig.add_trace(go.Bar(name='Average sentence length', x=c_names, y=average_sentence_lengths), row=2, col=2)

        fig.update_layout(title_text=title)
        fig.show()