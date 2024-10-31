from src.settings import Settings
import re

class Cleaner:

    def __init__(self, settings: Settings) -> None:
        self.configuration = settings.configuration
    
    def _is_too_small(self, text: str) -> bool:
        """Check if the text is too small"""
        words = text.split()
        return len(words) < self.configuration.min_response_number_of_words
    
    def _ends_with_repeated_substring(self, text: str) -> bool:
        """Check if the end of the text contains repeated substrings."""
        n = len(text)

        # Loop through possible lengths of repeating substrings
        for length in range(self.configuration.min_repeat_size, n // 2 + 1):  # length of substring
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
    def _remove_ats(text: str) -> str:
        """Replace all @ signs by the space character."""
        return text.replace("@", " ")
    
    @staticmethod
    def _remove_html_tags(text: str) -> str:
        """Remove HTML tags from the text"""
        pattern = r'<.*?>'
        return re.sub(pattern, '', text, flags=re.DOTALL)
    

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
    
    @staticmethod
    def _remove_italic(text: str) -> str:
        """Remove italic text from the text"""
        pattern = r'_(.*?)_'
        return re.sub(pattern, r'\1', text, flags=re.DOTALL)
    
    @staticmethod
    def _remove_dividers(text: str) -> str:
        """Remove dividers from the text"""
        pattern = r'(\s*\*\s*)+'
        return re.sub(pattern, '', text)

    @staticmethod
    def _remove_illustration_annotations(text: str) -> str:
        """Remove dividers from the text"""
        pattern = r'\[Illustration:.*?\]'
        return re.sub(pattern, '', text, flags=re.DOTALL)
    
    @staticmethod
    def _remove_note_annotation(text: str) -> str:
        """Remove note annotations from the text"""
        pattern = r'\{.*?\}'
        return re.sub(pattern, '', text)