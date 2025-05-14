from .abstractDataset import AbstractDataset
from datasets import load_dataset

class NEWSDataset(AbstractDataset):
    @property
    def hf_path(self):
        return "Themira/en_si_news_classification_with_label_name"

    @property
    def training_languages(self):
        return ["en"]

    @property
    def test_languages(self):
        return ["en", "si"]
    
    def get_train_dataset(self, lang):
        assert lang in self.training_languages, f"Language {lang} not supported for training."
        return load_dataset(self.hf_path, f"train_{lang}")

    def get_test_dataset(self, lang):
        assert lang in self.test_languages, f"Language {lang} not supported for testing."
        return load_dataset(self.hf_path, f"train_{lang}")

    def get_label_mapping(self):
        return dict()

    def process_prompt(self, example):
        sentence = f"News Sentence: {example['sentence']} Category: "
        label = example["label"]
        return sentence, label