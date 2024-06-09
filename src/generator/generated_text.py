from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
import json
from pathlib import Path

class GeneratedText:
    
    def __init__(self,
                 text: str,
                 requested_number_of_words: int,
                 model_name: str,
                 author_name: str,
                 prompt_template: ChatPromptTemplate,
                 query: str):
            self.data = {}
            self.data['id'] = int(datetime.now().timestamp())  
            self.data["requested_number_of_words"]=requested_number_of_words
            self.data["response_length"]=len(text)
            self.data["model"]=model_name
            self.data["created_at"]=datetime.now().isoformat()
            self.data["author"]=author_name
            self.data["prompt_template"]=str(prompt_template)
            self.data["query"]=query
            self.data["response"]=text
    
    def save(self, res_path: str):
        """Save generated text to a file"""
        filename = str(self.data["id"]) + ".json"
        path = Path(res_path) / self.data["model"] / self.data["author"] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.data, indent=4))