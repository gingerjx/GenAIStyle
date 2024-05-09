from src.author.text import Text
import json

class LLMResponse(Text):
    
    def __init__(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            response = json.load(f)
        self.text = response["response"]
        self.id = id
        self.prompt = response["prompt"]
        self.system_spec = response["system_spec"]
        self.created_at = response["created_at"]
        self.model = response["model"]

    def text(self):
        return self.text()
        