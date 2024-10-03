import numpy as np
import plotly.graph_objects as go

from src.analysis.visualization.analysis_visualization import AnalysisVisualization
from src.classification.classification_data import LogisticRegressionResults

class ClassificationVisualization(AnalysisVisualization):

    @staticmethod
    def visualize(classification_results: LogisticRegressionResults):
        model = classification_results.model
        X = classification_results.X
        y = classification_results.y

        color_map = {'human': 'blue', 'llm': 'red'}
        y_mapped = y.map(color_map)
        
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))


        # Predict probabilities for the meshgrid
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Create a 3D scatter plot
        fig = go.Figure()
        for target_name in ['human', 'llm']:
            fig.add_trace(go.Scatter(
                x=X[y == target_name].iloc[:, 0], 
                y=X[y == target_name].iloc[:, 1], 
                mode='markers', 
                marker=dict(color=AnalysisVisualization.HUMAN_LLM_COLORS[target_name]),
                name=target_name
            ))


        # Create a 3D surface plot for the decision boundary
        fig.add_trace(go.Scatter(
            x=xx[0], 
            y=yy[:, 0], 
            mode='lines', 
            line=dict(color='black'),
            name='Decision Boundary'
        ))


        # Customize the plot
        fig.update_layout(
            title='Logistic Regression Decision Boundary', 
            xaxis_title='PC1', 
            yaxis_title='PC2'
        )

        # Show the plot
        fig.show()
                