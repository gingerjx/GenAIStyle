import json
from pathlib import Path
from src.settings import Settings
import re
import os

class Cleaner:

    def __init__(self, settings: Settings) -> None:
        self.paths = settings.paths
        self.configuration = settings.configuration
        self.books_dir_name = settings.paths.raw_books_dir.parts[-1]
        
    def cleaned_generated_corpus_exists(self) -> bool:
        """Check if the cleaned corpus exists """
        return all(Path(self.paths.cleaned_models_dir, model_name).exists() 
                   for model_name 
                   in self.paths.raw_models_dirs.keys())

    def cleaned_books_corpus_exists(self) -> bool:
        """Check if the cleaned corpus exists"""
        return Path(self.paths.cleaned_books_dir).exists()

    def clean_generated_corpus(self) -> None:
        """Clean the generated corpus"""
        for model_name, model_dir in self.paths.raw_models_dirs.items():
            for file in model_dir.rglob("*"):
                if file.is_file():
                    with open(file, "r", encoding="utf-8") as f:
                        json_content = json.load(f)

                    cleaned_text = self._clean_text(json_content["response"])
                    if cleaned_text is None:
                        continue
                    
                    json_content["response"] = cleaned_text

                    cleaned_filepath = Path(
                        self.paths.cleaned_dir,
                        os.path.relpath(file, self.paths.raw_dir)
                    )
                    cleaned_filepath.parent.mkdir(parents=True, exist_ok=True)

                    with open(cleaned_filepath, "w", encoding="utf-8") as f:
                        json.dump(json_content, f)

    def clean_books_corpus(self) -> None:
        """Clean the books corpus"""
        for file in self.paths.raw_books_dir.rglob("*"):
            if file.is_file():
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()       
                # No cleaning needed        
                cleaned_filepath = Path(
                    self.paths.cleaned_dir,
                    os.path.relpath(file, self.paths.raw_dir)
                )
                cleaned_filepath.parent.mkdir(parents=True, exist_ok=True)

                with open(cleaned_filepath, "w", encoding="utf-8") as f:
                    f.write(text)


    def _clean_text(self, text: str) -> str:
        """Clean the text"""
        if self._is_too_small(text):
            return None
        if self._ends_with_repeated_substring(text):
            return None
        text = self._remove_emojis(text)
        return text

    def _is_too_small(self, text: str) -> bool:
        """Check if the text is too small"""
        words = text.split()
        return len(words) < self.configuration.min_response_text_length
    
    def _ends_with_repeated_substring(self, text: str) -> bool:
        """Check if the end of the text contains repeated substrings."""
        n = len(text)

        # Loop through possible lengths of repeating substrings
        for length in range(self.configuration.min_repeat_length, n // 2 + 1):  # length of substring
            # Get the last `length` characters as the candidate substring
            substring = text[-length:]
            count = 1
            j = n - length * 2  # Start checking before the last occurrence of the substring

            # Check for consecutive repeats from the end
            while j >= 0 and text[j:j+length] == substring:
                count += 1
                j -= length

            # If the count of repeats meets the threshold, return True
            if count >= self.configuration.repeat_threshold:
                return True
        
        return False
    
    @staticmethod
    def _remove_emojis(text: str) -> str:
        """Remove emojis from the text"""
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", re.UNICODE)
        return re.sub(emoj, '', text)