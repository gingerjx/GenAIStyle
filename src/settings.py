from pathlib import Path
from typing import Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    anthropic_api_key: str = Field(description="Anthropic API Key")
    google_api_key: str = Field(description="Google API Key") 
    mistral_api_key: str = Field(description="Mistral API Key")
    openai_api_key: str = Field(description="OpenAI API Key")

class Settings:
    
    class Paths:
        
        res_dir: Path = Path("res")
        
        metadata_dir: Path = res_dir / "metadata"
        query_filepath: Path = metadata_dir / "queries"
        all_authors_filepath: Path = metadata_dir / "all/author_list"
        all_books_csv_filepath: Path = metadata_dir / "all/books.csv"
        selected_authors_filepath: Path = metadata_dir / "selected/author_list"
        selected_books_csv_filepath: Path = metadata_dir / "selected/books.csv"

        raw_dir: Path = res_dir / "raw"
        raw_books_dir: Path = raw_dir / "books"
        raw_models_dir: Path = raw_dir / "models"
        raw_models_dirs: Dict[str, Path] = {
                "gpt-3.5-turbo-0125": raw_models_dir / "gpt-3.5-turbo-0125",
                "gpt-4o": raw_models_dir / "gpt-4o",
                "gemini-1.5-flash": raw_models_dir / "gemini-1.5-flash",
                "open-mixtral-8x7b": raw_models_dir / "open-mixtral-8x7b",
                "claude-3-haiku-20240307": raw_models_dir / "claude-3-haiku-20240307"
            }
        
        cleaned_dir: Path = res_dir / "cleaned"
        cleaned_books_dir: Path = cleaned_dir / "books"
        cleaned_models_dir: Path = cleaned_dir / "models"
        cleaned_models_dirs: Dict[str, Path] = {
                "gpt-3.5-turbo-0125": cleaned_models_dir / "gpt-3.5-turbo-0125",
                "gpt-4o": cleaned_models_dir / "gpt-4o",
                "gemini-1.5-flash": cleaned_models_dir / "gemini-1.5-flash",
                "open-mixtral-8x7b": cleaned_models_dir / "open-mixtral-8x7b",
                "claude-3-haiku-20240307": cleaned_models_dir / "claude-3-haiku-20240307"
            }
        
        results_dir: Path = res_dir / "results"
        analysis_filepath: Path = results_dir / "analysis.json"
          
    class Configuration:
        
        read_analysis_from_file: bool = False                           # Read analysis data from file
        response_number_of_words: int = 3000                            # Expected number of words used during response generation
        book_chunk_size: int = 5 * response_number_of_words             # Average english word length * expected number of words used during response generation
        analysis_size: int = 30000                                      # Number of words used during analysis   
        min_response_text_length: int = 100                             # Minimum number of words of the response text to be considered
        min_repeat_length: int = 3                                      # Minimum length of the repeated substring
        repeat_threshold: int = 3                                       # Minimum number of repeated substrings to be considered
        n_top_function_words: int = 10                                  # Number of top function words to be considered
    
    paths: Paths = Paths()
    configuration: Configuration = Configuration()



    