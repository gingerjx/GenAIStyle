from typing import List
import pandas as pd
from src.datasets.daigt.daigt_cleaner import DaigtCleaner
from src.models.collections.collection import Collection
from src.models.collections.daigt_collection import DaigtCollection
from src.settings import Settings


class DaigtDataset:
    
    def __init__(self, settings: Settings):
        self.configuration = settings.configuration
        self.paths = settings.paths
        self.cleaner = DaigtCleaner(settings)

        self.csv = None
        self.model_names = None
        self.raw_collections: List[Collection] = []
        self.cleaned_collections: List[Collection] = []

    def load(self) -> None:

        self.csv = pd.read_csv(self.paths.daigt_raw_dataset_filepath)
        self.collection_names = self.csv["model"].unique()

        for collection_name in self.collection_names:
            raw_collection = self.read_collection(collection_name)
            self.raw_collections.append(raw_collection)
            self.cleaned_collections.append(self.cleaner.clean(raw_collection))

    def read_collection(self, collection_name: str) -> DaigtCollection:
        collection = DaigtCollection(
            name=collection_name,
            seed=self.configuration.seed
        )
        collection.read(self.csv[self.csv["model"] == collection_name])
        return collection