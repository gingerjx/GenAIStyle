from pathlib import Path
from g4f.Provider import Gemini, Bing, DeepInfra, You

class Configuration():
    # Files
    RES_DIR = Path("res")

    ALL_BOOK_DIR = RES_DIR / "books/all"
    ALL_BOOK_DATA_DIR = ALL_BOOK_DIR / "data"
    ALL_AUTHORS_FILEPATH = ALL_BOOK_DIR / "author_list"
    ALL_BOOKS_CSV_FILEPATH = ALL_BOOK_DIR / "books.csv"

    SELECTED_BOOKS_DIR = RES_DIR / "books/selected"
    SELECTED_AUTHORS_FILEPATH = SELECTED_BOOKS_DIR / "author_list"
    SELECTED_AUTHORS_CSV_FILEPATH = SELECTED_BOOKS_DIR / "books.csv"

    # Models
    HAR_DIR = Path("hardir")
    GEMINI_COOKIES = HAR_DIR / "gemini_cookies.json"

    GEMINI_PRO_DATA_DIR = RES_DIR / "gemini-pro"
    GPT_4_DATA_DIR = RES_DIR / "gpt-4"
    GPT_3_5_TURBO_PRO_DATA_DIR = RES_DIR / "gpt-3.5-turbo"

    PROMPTS_FILEPATH = RES_DIR / "prompts"

    MODELS = [
        "gemini-pro",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    PROVIDERS = {
        "gemini-pro": Gemini,
        "mistralai/Mixtral-8x7B-Instruct-v0.1": DeepInfra,
        "gpt-4": Bing,
        "gpt-3.5-turbo": You    
    }
    RESPONSE_LENGTH = 5000