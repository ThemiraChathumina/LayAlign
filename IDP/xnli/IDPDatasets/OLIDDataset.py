from .abstractDataset import AbstractDataset
from datasets import load_dataset

class OLIDDataset(AbstractDataset):
    @property
    def hf_path(self):
        return "christophsonntag/OLID"

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
            "OFF": "Offensive",
            "NOT": "Not Offensive"
        }

    def process_prompt(self, example):
        sentence = f"Tweet: {example['tweet']} Category: "
        label = self.labels[example["subtask_a"].strip()]
        return sentence, label