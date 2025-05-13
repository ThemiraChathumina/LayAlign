from .abstractDataset import AbstractDataset

class XNLIDataset(AbstractDataset):
    @property
    def hf_path(self):
        return "facebook/xnli"

    @property
    def training_languages(self):
        return ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

    @property
    def test_languages(self):
        return ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

    def get_label_mapping(self):
        return {
            2: "Contradiction",
            0: "Entailment",
            1: "Neutral"
        }

    def process_prompt(self, example):
        sentence = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']} Label: "
        label = self.labels[int(example["label"])]
        return sentence, label