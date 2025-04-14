import json
from torch.utils.data import Dataset
import random
from datasets import load_dataset


langs_map = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
                     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
                     'Russian': 'ru', 'Thai': 'th','Telugu': 'te', 'Greek': 'el',
                     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
                     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
                     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
                     'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur'}

class Data(Dataset):
    def __init__(self, dataset_path, split):
        super().__init__()
        if 'json' in dataset_path:
            self.ds = load_dataset(
                'json',
                data_files=dataset_path)['train']
        else:
            self.ds = load_dataset(dataset_path)[split]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample['source']=sample['input']
        sample['target']=sample['output'],
        sample['source_language']='English',
        sample['target_language']='English',
        sample['prompt']=sample['input'],
        return sample


def construct_prompt_xnli(sample):
    sentence1 = sample['sentence1']
    sentence2 = sample['sentence2']
    source = f'Premise: {sentence1}\nHypothesis: {sentence2}\nLabel:'
    return source

def construct_prompt_x_csqa(sample):
    question = sample['question']['stem']
    choices = sample['question']['choices']
    choice_text = [chr(ord('A') + i) + '. ' + choice['text'] for i, choice in enumerate(choices)]
    source = question + '\nOptions:' + '\t'.join(choice_text) + '\nAnswer:'
    return source


def construct_prompt_math(query):
    # return query
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{query}\n\n### Response: Let's think step by step."
    )
    return prompt_no_input


def construct_prompt_mt(query, src_lang_name, trg_lang_name):
    instruction = f'Translate the following sentences from {src_lang_name} to {trg_lang_name}.'
    prompt_no_input = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt_no_input



class MathDataset(Dataset):
    def __init__(self, dataset, task) -> None:
        super().__init__()
        self.dataset = dataset
        self.task = task
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.task == 'translation':
            sample['prompt'] = construct_prompt_mt(sample['source'],
                                                   sample['source_language'],
                                                   sample['target_language'])
        elif 'csqa' in self.task:
            sample['source'] = sample['prompt'] = construct_prompt_x_csqa(sample)
        elif 'nli' in self.task:
            sample['source'] = sample['prompt'] = construct_prompt_xnli(sample)
        else:
            sample['prompt'] = construct_prompt_math(sample['source'])
        return sample

def read_lego(train_num, languages):
    dataset_train = []
    for train_name in languages:
        train_name_map = langs_map[train_name]
        path_base = f'./datas/bilingual_pairs/en-{train_name_map}'
        path_src = f'{path_base}/train_100k.{train_name_map}'
        path_trg = f'{path_base}/train_100k.en'
        sources = read_dataset(path_src)[:train_num]
        targets = read_dataset(path_trg)[:train_num]
        train_set = [(source, target) for source, target in zip(sources, targets)]
        for source, target in train_set:
            dataset_train.append({
                'source': source,
                'target': target,
                'source_language': train_name,
                'target_language': 'English'
            })
    random.shuffle(dataset_train)
    return dataset_train


def read_x_csqa_train():
    dataset_names = ['Urdu', 'Swahili', 'Hindi', 'Arabic', 'Vietnamese', 'Japanese', 'Polish',
                     'Chinese', 'Flemish', 'Russian', 'Italian', 'German', 'Portuguese', 'French',
                     'Spanish', 'English']
    dataset_train, dataset_dev, dataset_test = [], [], []
    for dataset_name in dataset_names:
        name_abb = langs_map[dataset_name]
        train_set = read_dataset(f'./datas/query_translation/x-csqa/train_{name_abb}.jsonl')
        for sample in train_set:
            sample['source_language'] = dataset_name
            sample['target_language'] = 'English'
            sample['target'] = sample['answerKey']
            dataset_train.append(sample)
    random.shuffle(dataset_train)
    return dataset_train


def read_xnli_train():
    ds = load_dataset("facebook/xnli", "en")
    train_ds = ds['train']
    train_set = []
    labels = {0:'entailment', 1:'neutral', 2:'contradiction'}
    for i in range(30000):
        sample = {}
        sample['id'] = i
        sample['language'] = 'English'
        sample['sentence1'] = train_ds[i]['premise']
        sample['sentence2'] = train_ds[i]['hypothesis']
        sample['label'] = f"{labels[int(train_ds[i]['label'])]}"
        sample['target'] = f"{labels[int(train_ds[i]['label'])]}"
        sample['source_language'] = 'English'
        sample['target_language'] = 'English'
        train_set.append(sample)
    random.shuffle(train_set)
    return train_set


def read_math_train(train_num):
    languages = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian', 'Spanish']
    train_set = read_dataset('./datas/query_translation/metamath_615k.json')
    train_sets = {}
    for sample in train_set:
        lang = sample['lang']
        sample = {
            'source': sample['query'],
            'target': sample['response'],
            'source_language': lang,
            'target_language': 'English'
        }
        if lang not in train_sets:
            train_sets[lang] = [sample]
        else:
            if len(train_sets[lang]) < train_num:
                train_sets[lang].append(sample)
    dataset_train = []
    for lang in train_sets:
        if lang in languages:
            dataset = train_sets[lang]
            for sample in dataset:
                dataset_train.append(sample)
    random.shuffle(dataset_train)
    return dataset_train


def read_metamath_200k(dataset_path, split):
    train_set = load_dataset(dataset_path)[split]
    train_sets = {}
    for sample in train_set:
        lang = 'English'
        sample = {
            'source': sample['input'],
            'target': sample['output'],
            'source_language': lang,
            'target_language': 'English'
        }
        if lang not in train_sets:
            train_sets[lang] = [sample]
        else:
            train_sets[lang].append(sample)
    dataset_train = []
    for lang in train_sets:
        dataset = train_sets[lang]
        for sample in dataset:
            dataset_train.append(sample)
    random.shuffle(dataset_train)
    return dataset_train


def read_msvamp():
    datasets_test = {}
    test_list = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                 'Spanish', 'English']
    for test_name in test_list:
        test_path = f'./datas/evaluation/msvamp/test_{test_name}.jsonl'
        dataset = read_dataset(test_path)
        dataset_test = []
        for sample in dataset:
            dataset_test.append({
                'source': sample['m_query'],
                'target': sample['response'],
                'source_language': test_name,
                'target_language': test_name
            })
        datasets_test[test_name] = dataset_test
    return datasets_test


def read_docs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    return lines

def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset


def read_mgsms():
    datasets_test = {}
    test_list = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                 'Spanish', 'English']
    for test_name in test_list:
        test_name_abb = langs_map[test_name]
        test_path = f'./datas/evaluation/mgsm/mgsm_{test_name_abb}.tsv'
        dataset = read_mgsm(test_path, test_name)
        dataset_test = []
        for sample in dataset:
            dataset_test.append({
                'source': sample['question'],
                'target': sample['answer'],
                'source_language': test_name,
                'target_language': test_name
            })
        datasets_test[test_name] = dataset_test
    return datasets_test

def read_mgsm(path, language):
    dataset = read_docs(path)
    dataset_new = []
    for i, sample in enumerate(dataset):
        sp = sample.split("\t")
        question = sp[0]
        answer = sp[1].replace(",", '').strip()
        explanation = ""
        dataset_new.append({
            'id': i,
            'question': question,
            'explanation': explanation,
            'answer': answer,
            'language': language,
            'source_language': language,
            'target_language': language
        })
    return dataset_new


def read_x_csqa():
    dataset_names = ['Urdu', 'Swahili', 'Hindi', 'Arabic', 'Vietnamese', 'Japanese', 'Polish',
                     'Chinese', 'Flemish', 'Russian', 'Italian', 'German', 'Portuguese', 'French',
                     'Spanish', 'English']
    dataset_tests = {}
    for dataset_name in dataset_names:
        name_abb = langs_map[dataset_name]
        dataset = read_dataset(f'./datas/evaluation/x-csqa/{name_abb}/dev.jsonl')
        dataset_test = []
        for sample in dataset:
            sample['source_language'] = dataset_name
            sample['target_language'] = 'English'
            sample['target'] = sample['answerKey']
            dataset_test.append(sample)
        dataset_tests[dataset_name] = dataset_test
    return dataset_tests



def read_xnli():
    dataset = read_dataset(f'/root/LayAlign/xnli_layalign/test.json')
    test_sets = {}
    for sample in dataset:
        sample['target'] = sample['label']
        language = sample['language']
        sample['source_language'] = sample['language']
        sample['target_language'] = 'English'
        if language not in test_sets:
            test_sets[language] = [sample]
        else:
            test_sets[language].append(sample)
    return test_sets



def read_flores():
    flores_langs_map = {'Swahili': 'swh', 'Bengali': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
                        'German': 'deu', 'Spanish': 'spa', 'French': 'fra', 'Japanese': 'jpn', 'Russian': 'rus', }
    test_list = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese',
                 'German', 'Spanish', 'French', 'Russian']
    datasets_test = {}
    for test_name in test_list:
        dataset = []
        flores_name = flores_langs_map[test_name]
        path_base = f'./datas/flores101/devtest/'
        path_src = f'{path_base}/{flores_name}.devtest'
        flores_name = flores_langs_map['English']
        path_trg = f'{path_base}/{flores_name}.devtest'
        sources = read_dataset(path_src)
        targets = read_dataset(path_trg)
        for source, target in zip(sources, targets):
            dataset.append({
                'source': source,
                'target': target,
                'source_language': test_name,
                'target_language': 'English'
            })
        datasets_test[test_name] = dataset
    return datasets_test