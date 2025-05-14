from .abstractDataset import AbstractDataset
from datasets import load_dataset

class SISENDataset(AbstractDataset):
    @property
    def hf_path(self):
        return "sinhala-nlp/sinhala-sentiment-analysis"

    @property
    def training_languages(self):
        return ["si"]

    @property
    def test_languages(self):
        return ["si"]
    
    def get_train_dataset(self, lang):
        assert lang in self.training_languages, f"Language {lang} not supported for training."
        return load_dataset(self.hf_path, f"train")

    def get_test_dataset(self, lang):
        assert lang in self.test_languages, f"Language {lang} not supported for testing."
        return load_dataset(self.hf_path, f"test")

    def get_label_mapping(self):
        return {
            "POSITIVE": "Positive",
            "NEGATIVE": "Negative",
            "NEUTRAL": "Neutral"
        }

    def process_prompt(self, example):
        sentence = f"Text: {example['comment_phrase']} Label: "
        label = self.labels[example["comment_sentiment"].strip()]
        return sentence, label