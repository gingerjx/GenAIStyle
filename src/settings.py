from pathlib import Path
from typing import Dict, List
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

    all_book_dir: Path = Field(Path("res/books/all"),   
                               description="Path to the directory containing all books")
    all_book_data_directory: Path = Field(Path("res/books/all/data"), 
                                     description="Path to the directory containing all data")
    all_authors_filepath: Path = Field(Path("res/books/all/author_list"), 
                                       description="Path to the file containing all authors")
    all_books_csv_filepath: Path = Field(Path("res/books/all/books.csv"), 
                                         description="Path to the file containing all books")
    
    selected_books_dir: Path = Field(Path("res/books/selected"), 
                                     description="Path to the directory containing selected books")
    selected_authors_filepath: Path = Field(Path("res/books/selected/author_list"), 
                                        description="Path to the file containing selected authors")
    selected_books_csv_filepath: Path = Field(Path("res/books/selected/books.csv"), 
                                             description="Path to the file containing selected books")
    
    response_length: int = Field(3000, 
                                 description="Length of the response")
    max_tokens: int = Field(4096,
                            description="Maximum tokens of LLM output")
    analysis_size: int = Field(10000,
                                 description="Size of the analysis")
    model_dirs: Dict[str, Path] = Field({
        "gpt-3.5-turbo-0125": Path("res/gpt-3.5-turbo-0125"),
        "gpt-4o": Path("res/gpt-4o"),
        "gemini-1.5-flash": Path("res/gemini-1.5-flash"),
        "open-mixtral-8x7b": Path("res/open-mixtral-8x7b"),
        "claude-3-haiku-20240307": Path("res/claude-3-haiku-20240307")
        # "geminipro": Path("res/gemini-pro"),
        # "gpt-4": Path("res/gpt-4"),
        # "gpt-3.5-turbo": Path("res/gpt-3.5-turbo")
    }, description="List of models' directories")