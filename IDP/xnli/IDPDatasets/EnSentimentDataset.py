from .abstractDataset import AbstractDataset
from datasets import load_dataset

class ENSENDataset(AbstractDataset):
    @property
    def hf_path(self):
        return "Sp1786/multiclass-sentiment-analysis-dataset"

    @property
    def training_languages(self):
        return ["en"]

    @property
    def test_languages(self):
        return ["en"]
    
    def get_train_dataset(self, lang):
        assert lang in self.training_languages, f"Language {lang} not supported for training."
        return load_dataset(self.hf_path, f"train")

    def get_test_dataset(self, lang):
        assert lang in self.test_languages, f"Language {lang} not supported for testing."
        return load_dataset(self.hf_path, f"test")

    def get_label_mapping(self):
        return {
            2: "Positive",
            0: "Negative",
            1: "Neutral"
        }

    def process_prompt(self, example):
        sentence = f"Text: {example['text']} Label: "
        label = self.labels[int(example["label"])]
        return sentence, label