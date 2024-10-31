from typing import Dict, List

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
    HUMAN_LLM_COLORS = {
        "human": "#3498db", 
        "llm": "#e74c3c"
    }
    PN_COLORS = {
        "positive": "#3498db", 
        "negative": "#e74c3c"
    }

    HUMAN_COLORS = {
        "human": "#3498db", 
        "books": "#3498db",
    }
    COLORS = [
        "#e74c3c",
        "#2ecc71",
        "#f1c40f",
        "#9b59b6",
        "#e67e22",
        "#1abc9c",
        "#34495e",
        "#d35400",
        "#7f8c8d",
        "#27ae60",
        "#8e44ad",
    ]

    @staticmethod
    def get_color_dict(collection_names: List[str]) -> Dict[str, str]:
        color_dict = {
            collection_name: AnalysisVisualization.HUMAN_COLORS[collection_name] 
            for collection_name in collection_names
            if collection_name in AnalysisVisualization.HUMAN_COLORS.keys()
        }
        model_names = [
            name 
            for name in collection_names 
            if name not in AnalysisVisualization.HUMAN_COLORS.keys()
        ]
        
        for i, model_name in enumerate(model_names):

            color_dict[model_name] = AnalysisVisualization.COLORS[i]
        
        return color_dict#