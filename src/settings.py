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
    
    response_length: int = Field(5000, 
                                 description="Length of the response")