from abc import ABC, abstractmethod
from datasets import load_dataset

from collections import defaultdict

class AbstractDataset(ABC):
    def __init__(self):
        self.labels = self.get_label_mapping()

    @property
    @abstractmethod
    def hf_path(self):
        """Hugging Face dataset path (e.g., 'mteb/amazon_reviews_multi')"""
        pass

    @property
    @abstractmethod
    def training_languages(self):
        pass

    @property
    @abstractmethod
    def test_languages(self):
        pass

    @abstractmethod
    def get_label_mapping(self):
        """Returns the label dictionary."""
        pass

    @abstractmethod
    def process_prompt(self, example):
        """Takes an example and returns (sentence, label)"""
        pass

    def get_train_dataset(self, lang):
        assert lang in self.training_languages, f"Language {lang} not supported for training."
        return load_dataset(self.hf_path, lang)["train"]

    def get_test_dataset(self, lang):
        assert lang in self.test_languages, f"Language {lang} not supported for testing."
        return load_dataset(self.hf_path, lang)["test"]

    def get_processed_set(self, dataset, limit=None):
        if limit is not None:
            # Assume you want limit examples TOTAL, divided evenly across classes
            num_classes = len(set(dataset.unique('label')))
            limit_per_class = max(limit // num_classes, 1)
            dataset = self.stratified_select(dataset, limit_per_class=limit_per_class)
        return [
            {"prompt": sentence, "target": label}
            for example in dataset
            for sentence, label in [self.process_prompt(example)]
        ]

    def get_train_set(self, lang, limit=None):
        return self.get_processed_set(self.get_train_dataset(lang), limit=limit)

    def get_test_set(self, lang, limit=None):
        return self.get_processed_set(self.get_test_dataset(lang), limit=limit)

    def get_test_sets(self, limit=None):
        return {lang: self.get_test_set(lang, limit=limit) for lang in self.test_languages}

    def stratified_select(self,dataset, label_column='label', limit_per_class=100):
        label_to_indices = defaultdict(list)
        for i, item in enumerate(dataset):
            label_to_indices[item[label_column]].append(i)

        selected_indices = []
        for label, indices in label_to_indices.items():
            selected_indices.extend(indices[:limit_per_class])

        return dataset.select(selected_indices).shuffle(seed=42)
