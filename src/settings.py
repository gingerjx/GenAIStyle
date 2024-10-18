from pathlib import Path
from typing import Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Secrets(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(env_file=".env")

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
        selected_dir_path = metadata_dir / "selected"
        selected_authors_filepath: Path = selected_dir_path / "author_list"
        selected_books_csv_filepath: Path = selected_dir_path / "books.csv"

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
        
        read_analysis_from_file: bool = False                                                       # Read analysis data from file
        response_number_of_words: int = 3000                                                        # Expected number of words used during response generation
        extract_book_chunk_size: int = 5 * response_number_of_words                                 # Number of characters used as a chunk size during preprocessing of the books. Average english word length * expected number of words used during response generation
        analysis_chunk_number_of_words: int = 5000                                                  # Number of words used as a chunk size during analysis. Has to be divisor of `analysis_number_of_words`
        analysis_number_of_words: int = 200000                                                      # Number of words used during analysis   
        analysis_number_of_chunks: int = analysis_number_of_words // analysis_chunk_number_of_words # Number of chunks used during analysis
        min_response_number_of_words: int = 100                                                     # Minimum number of words of the response text to be considered
        min_repeat_size: int = 3                                                                    # Minimum length of the repeated substring
        repeat_threshold: int = 3                                                                   # Minimum number of repeated substrings to be considered
        top_n_function_words: int = 10                                                              # Number of top function words to be considered
        test_size: float = 0.2                                                                      # Test size for train-test split
        seed: int = 42                                                                              # Seed for random state
        number_of_cv_folds: int = 5                                                                 # Number of cross-validation folds
    
    paths: Paths = Paths()
    configuration: Configuration = Configuration()



    