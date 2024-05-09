import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

class Analysis():

    @staticmethod
    def merge_generated_text(authors_generated_text: dict) -> str:
        merged_text = {}
        for author_name, generated_texts in authors_generated_text.items():
            merged_text[author_name] = " ".join([data["response"] for data in generated_texts])
        return merged_text

    @staticmethod
    def get_text_analysis(text: str) -> dict:
        analysis = {}

        words = text.split()
        analysis["word_count"] = len(words)
        analysis["unique_word_count"] = len(set(words))
        analysis["average_word_length"] = sum(len(word) for word in words) / len(words)
        
        sentences = sent_tokenize(text)
        analysis["average_sentence_length"] = len(sentences) / len(words)

        return analysis

    @staticmethod
    def analyse_generated_text(authors_generated_text: dict, size: int = None) -> dict:
        merged_text = Analysis.merge_generated_text(authors_generated_text)
        analysis = {}
        for author_name, text in merged_text.items():
            if size:
                text = text[:size]
            analysis[author_name] = Analysis.get_text_analysis(text)
        return analysis