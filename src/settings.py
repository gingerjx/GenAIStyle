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
        
        # Writing Style Dataset
        
        ws_dir: Path = res_dir / "datasets/writing_style"
        ws_metadata_dir: Path = ws_dir / "metadata"
        ws_query_filepath: Path = ws_metadata_dir / "queries"
        ws_all_authors_filepath: Path = ws_metadata_dir / "all/author_list"
        ws_all_books_csv_filepath: Path = ws_metadata_dir / "all/books.csv"
        ws_selected_dir_path = ws_metadata_dir / "selected"
        ws_selected_authors_filepath: Path = ws_selected_dir_path / "author_list"
        ws_selected_books_csv_filepath: Path = ws_selected_dir_path / "books.csv"

        ws_raw_dir: Path = ws_dir / "raw"
        ws_raw_books_dir: Path = ws_raw_dir / "books"
        ws_raw_models_dir: Path = ws_raw_dir / "models"
        ws_raw_models_dirs: Dict[str, Path] = {
                "gpt-3.5-turbo-0125": ws_raw_models_dir / "gpt-3.5-turbo-0125",
                "gpt-4o": ws_raw_models_dir / "gpt-4o",
                "gemini-1.5-flash": ws_raw_models_dir / "gemini-1.5-flash",
                "open-mixtral-8x7b": ws_raw_models_dir / "open-mixtral-8x7b",
                "claude-3-haiku-20240307": ws_raw_models_dir / "claude-3-haiku-20240307"
            }

        # Daigt Dataset

        daigt_dir: Path = res_dir / "datasets/daigt"

        daigt_raw_dir: Path = daigt_dir / "raw"
        daigt_raw_dataset_filepath: Path = daigt_raw_dir / "train_v4_drcat_01.csv"
          
    class Configuration:

        seed: int = 42                                                                              # Seed for random state

        # Analysis
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
        top_n_punctuation: int = 5                                                                  # Number of top punctuation to be considered

        # Classification
        training_max_iter: int = 100                                                                # Maximum number of iterations for the logistic regression
        test_size: float = 0.2                                                                      # Test size for the logistic regression
        number_of_cv_folds: int = 5                                                                 # Number of cross-validation folds for the logistic regression
    
    paths: Paths = Paths()
    configuration: Configuration = Configuration()



    