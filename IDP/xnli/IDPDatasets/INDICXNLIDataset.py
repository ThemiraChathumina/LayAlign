from XNLIDataset import XNLIDataset

class INDICXNLIDataset(XNLIDataset):
    @property
    def hf_path(self):
        return "Divyanshu/indicxnli"

    @property
    def training_languages(self):
        return ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

    @property
    def test_languages(self):
        return ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
