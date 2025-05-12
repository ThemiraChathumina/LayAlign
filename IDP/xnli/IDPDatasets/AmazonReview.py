from .abstractDataset import AbstractDataset

class AmazonReviewDataset(AbstractDataset):
    @property
    def hf_path(self):
        return "mteb/amazon_reviews_multi"

    @property
    def training_languages(self):
        return ["en", "es", "fr", "de", "zh", "ar"]

    @property
    def test_languages(self):
        return ["en", "es", "fr", "de", "zh", "ar"]

    def get_label_mapping(self):
        return {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Very Positive"
        }

    def process_prompt(self, example):
        sentence = f"Review: {example['text']} Label: "
        label = self.labels[int(example["label"])]
        return sentence, label
