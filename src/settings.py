from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    anthropic_api_key: str = Field(description="Anthropic API Key")
    google_api_key: str = Field(description="Google API Key") 
    mistral_api_key: str = Field(description="Mistral API Key")
    openai_api_key: str = Field(description="OpenAI API Key")

class Configuration(BaseModel):
    
    res_directory: Path = Field(Path("res"), 
                                description="Path to the resources directory")
    query_filepath: Path = Field(Path("res/queries"), 
                                 description="Path to the file containing queries")
    selected_authors_filepath: Path = Field(Path("res/books/selected/author_list"), 
                                            description="Path to the file containing selected authors")
    response_length: int = Field(5000, 
                                 description="Length of the response")