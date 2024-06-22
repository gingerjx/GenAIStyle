from typing import Dict, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.generator.generated_text import GeneratedText

class TextGenerator:
    
    def __init__(self, 
                 models: Dict[str, BaseChatModel],
                 queries_path: str,
                 authors_path: str,
                 res_directory: str,
                 response_number_of_words: int = 5000):
        self.models = models
        self.authors = open(authors_path, 'r', encoding='utf-8').read().split('\n')
        self.queries = open(queries_path, 'r', encoding='utf-8').read().split('\n')
        self.res_directory = res_directory
        self.response_number_of_words = response_number_of_words

    def generate(self) -> List[GeneratedText]:
        """Generate and save texts for all models, authors and queries"""
        generated_texts = []
        unprocessed_requests = []
        i = 1
        total = len(self.authors) * len(self.queries) * len(self.models)
        for author_name in self.authors:
            for query in self.queries:
                for model_name, model in self.models.items():
                    try:
                        print(f"[{i}/{total}] Generating [{model_name}]-[{author_name}] {query}")
                        prompt_template = self._get_prompt_template()
                        generated_text = self._generate_internal(model, 
                                                                prompt_template, 
                                                                query,
                                                                author_name)
                        genereated_text_transformed = self._transform(generated_text.content,
                                            model_name,
                                            author_name,
                                            query
                        )
                        genereated_text_transformed.save(self.res_directory)
                        generated_texts.append(genereated_text_transformed)
                        i += 1
                    except Exception as e:
                        print(f"Generation failed due to: {e}")
                        unprocessed_requests.append((model_name, author_name, query))
        return generated_texts, unprocessed_requests
    
    def _generate_internal(self, 
                           model: BaseChatModel, 
                           prompt_template: ChatPromptTemplate, 
                           query: str, 
                           author_name: str):
        """Generate text using model and prompt template"""
        chain = prompt_template | model
        return chain.invoke(
            {
                "author": author_name,
                "response_number_of_words": self.response_number_of_words,
                "query": query
            }
        )
    
    def _get_prompt_template(self):
        """Get prompt template for the text generation"""
        return ChatPromptTemplate.from_messages([
            ("system", 
            "Come up with the answer in {author}'s writing style. Don't use direct references and citations of {author}. Answer in plain text format. Use {response_number_of_words} words."),
            ("human", 
            "{query}"),
        ])
    
    def _transform(self, 
                   generated_text: str, 
                   model_name: str, 
                   author_name: str,
                   query: str) -> GeneratedText:
        """Transform generated text to the GeneratedText format"""
        return GeneratedText(
            text=generated_text,
            requested_number_of_words=self.response_number_of_words,
            model_name=model_name,
            author_name=author_name,
            prompt_template=self._get_prompt_template(),
            query=query
        )