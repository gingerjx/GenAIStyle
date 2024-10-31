from src.datasets.common.texts.text import Text
import json

class LLMResponse(Text):

    def __init__(self, text: str, id: str, query: str, prompt_template: str, created_at: str, model: str, filepath: str):
        self.text = text
        self.id = id
        self.query = query
        self.prompt_template = prompt_template
        self.created_at = created_at
        self.model = model
        self.filepath = filepath

    def get_text(self) -> str:
        """Get the text of the response"""
        return self.text
    
    def set_text(self, text: str) -> None:
        """Set the text of the response"""
        self.text = text
    
    def copy(self):
        """Return a copy of the response"""
        return LLMResponse(
            text=None,
            id=self.id,
            query=self.query,
            prompt_template=self.prompt_template,
            created_at=self.created_at,
            model=self.model,
            filepath=self.filepath
        )
    
    def from_file(filepath: str) -> 'LLMResponse':
        """Create a response from a file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            response = json.load(f)
        return LLMResponse(
            text=response["response"],
            id=response["id"],
            query=response["query"],
            prompt_template=response["prompt_template"],
            created_at=response["created_at"],
            model=response["model"],
            filepath=filepath
        )