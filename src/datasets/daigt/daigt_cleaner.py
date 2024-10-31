from src.datasets.common.cleaner import Cleaner
from src.datasets.daigt.texts.daigt_text import DaigtText
from src.datasets.daigt.collections.daigt_collection import DaigtCollection


class DaigtCleaner(Cleaner):
    
    def clean(self, daigt_collection: DaigtCollection) -> DaigtCollection:
        """Clean the Daigt collection"""
        cleaned_collection = DaigtCollection(daigt_collection.name, daigt_collection.seed)
        for daigt_text in daigt_collection.texts:
            if self._ends_with_repeated_substring(daigt_text.get_text()):
                continue
            cleaned_collection.texts.append(self._clean_text(daigt_text))
        return cleaned_collection
    
    def _clean_text(self, daigt_text: DaigtText) -> DaigtText:
        """Clean the text"""

        text = self._remove_emojis(daigt_text.get_text())
        text = Cleaner._remove_ats(text)
        text = Cleaner._remove_html_tags(text)
        return daigt_text.copy(text)
