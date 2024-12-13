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
        sessions_dir: Path = res_dir / "sessions"

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
        
        ws_dataset_dump_file = sessions_dir / "writing_style_dataset.db"
        ws_pca_classificaiton_dump_file = sessions_dir / "writing_style_pca_classification.db"
        ws_all_features_classification_dump_file = sessions_dir / "writing_style_all_features_classification.db"
        
        # Daigt Dataset

        daigt_dir: Path = res_dir / "datasets/daigt"

        daigt_raw_dir: Path = daigt_dir / "raw"
        daigt_raw_dataset_filepath: Path = daigt_raw_dir / "train_v4_drcat_01.csv"

        daigt_dataset_dump_file = sessions_dir / "daigt_dataset.db"
        daigt_pca_classificaiton_dump_file = sessions_dir / "daigt_pca_classification.db"

        # Twitter and Reddit Dataset

        tr_dir: Path = res_dir / "datasets/twitter_reddit"

        tr_raw_dir: Path = tr_dir / "raw"
        twitter_raw_dataset_filepath: Path = tr_raw_dir / "Twitter_Data.csv"
        reddit_raw_dataset_filepath: Path = tr_raw_dir / "Reddit_Data.csv"

        # News Dataset

        news_dir: Path = res_dir / "datasets/news"

        news_raw_dir: Path = news_dir / "raw"
        news_raw_dataset_filepath: Path = news_raw_dir / "data.csv"

        # Legal Dataset

        legal_dir: Path = res_dir / "datasets/legal"

        legal_raw_dir: Path = legal_dir / "raw"

    class Configuration:

        seed: int = 42                                                                              # Seed for random state
        min_response_number_of_words: int = 100                                                     # Minimum number of words of the response text to be considered
        min_repeat_size: int = 2                                                                    # Minimum length of the repeated substring
        repeat_threshold: int = 3                                                                   # Minimum number of repeated substrings to be considered
        top_n_function_words: int = 10                                                              # Number of top function words to be considered
        top_n_punctuation: int = 5                                                                  # Number of top punctuation to be considered

        # Writing Style Dataset Analysis
        ws_response_number_of_words: int = 3000                                                              # Expected number of words used during response generation
        ws_extract_book_chunk_size: int = 5 * ws_response_number_of_words                                    # Number of characters used as a chunk size during preprocessing of the books. Average english word length * expected number of words used during response generation
        ws_analysis_chunk_number_of_words: int = 5000                                                        # Number of words used as a chunk size during analysis. Has to be divisor of `ws_analysis_number_of_words`
        ws_analysis_number_of_words: int = 200000                                                            # Number of words used during analysis   
        ws_analysis_number_of_chunks: int = ws_analysis_number_of_words // ws_analysis_chunk_number_of_words # Number of chunks used during analysis
        ws_entropy_analysis_number_of_bins: int = 100                                                        # Number of bins used during entropy analysis

        # Daigt Dataset Analysis
        daigt_analysis_chunk_number_of_words: int = 5000                                                          # Number of words used as a chunk size during analysis. Has to be divisor of `daigt_analysis_number_of_words`
        daigt_analysis_number_of_words: int = None                                                                # Number of words used during analysis  

        # Classification
        training_max_iter: int = 100                                                                # Maximum number of iterations for the logistic regression
        test_size: float = 0.2                                                                      # Test size for the logistic regression
        number_of_cv_folds: int = 5                                                                 # Number of cross-validation folds for the logistic regression
    
    paths: Paths = Paths()
    configuration: Configuration = Configuration()



    