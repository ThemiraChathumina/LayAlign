from IDPDatasets import *

class IDPConfigs:
    def __init__(self):
        self.system_prompt = None
        self.baseline = True
        self.dataset = "xnli"
        self.improvement_under_test = "softmax"
        self.args = "SystemPromptAdded" if self.system_prompt else "NoArgs"
        self.run_name  = "baseline" if self.baseline else  self.improvement_under_test
        self.save_name = f"{self.dataset}_{self.run_name}"
        self.gate_csv_path = f"{self.save_name}.csv"
        self.lang = 'en'
        self.num_epochs = 3
        self.checkpoint = f'./outputs/{self.save_name}/epoch_{self.num_epochs-1}_{self.args}/pytorch_model.bin'
        self.dataset : AbstractDataset = AmazonReviewDataset()
        self.train_limit = 100000

    def getPremise(self):
        if self.lang == 'en':
            return "Premise"
        elif self.lang == 'es':
            return "Premisa"
        elif self.lang == 'fr':
            return "Prémisse"
        elif self.lang == 'de':
            return "Prämisse"
        elif self.lang == 'zh':
            return "前提"
        elif self.lang == 'ar':
            return "مقدمة"
        else:
            return "Premise"

    def getHypothesis(self):
        if self.lang == 'en':
            return "Hypothesis"
        elif self.lang == 'es':
            return "Hipótesis"
        elif self.lang == 'fr':
            return "Hypothèse"
        elif self.lang == 'de':
            return "Hypothese"
        elif self.lang == 'zh':
            return "假设"
        elif self.lang == 'ar':
            return "فرضية"
        else:
            return "Hypothesis"
    
    def getLabel(self):
        if self.lang == 'en':
            return "Label"
        elif self.lang == 'es':
            return "Etiqueta"
        elif self.lang == 'fr':
            return "Étiquette"
        elif self.lang == 'de':
            return "Etikett"
        elif self.lang == 'zh':
            return "标签"
        elif self.lang == 'ar':
            return "تسمية"
        else:
            return "Label"