import numpy as np
import plotly.graph_objects as go

from src.analysis.visualization.analysis_visualization import AnalysisVisualization
from src.classification.classification_data import ClassificationData, ClassificationResults

class ClassificationVisualization(AnalysisVisualization):

    @staticmethod
    def visualize_binary_logistic_regression_classification(classification_results: ClassificationData):
        model = classification_results.model
        X = classification_results.X
        y = classification_results.y
        
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        # Predict probabilities instead of labels for smoother contours
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Create a scatter plot for each class
        fig = go.Figure()

        for target_name in ['human', 'llm']:
            fig.add_trace(go.Scatter(
                x=X[y == target_name].iloc[:, 0], 
                y=X[y == target_name].iloc[:, 1], 
                mode='markers', 
                marker=dict(color=AnalysisVisualization.HUMAN_LLM_COLORS[target_name]),
                name=target_name,
                showlegend=True
            ))

        # Add the decision boundary as a single contour at probability 0.5
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.02),
            y=np.arange(y_min, y_max, 0.02),
            z=Z,
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                size=1,
                coloring='lines'
            ),
            colorscale=[
                [0.0, 'black'],
                [1.0, 'black']
            ],
            name='Decision Boundary',
            showlegend=True
        ))
        
        # Customize the plot
        fig.update_layout(
            title='Logistic Regression Decision Boundary for all chunks', 
            xaxis_title='PC1', 
            yaxis_title='PC2'
        )

        # Show the plot
        fig.show()
                