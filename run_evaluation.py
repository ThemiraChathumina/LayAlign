import torch.fx
from transformers import AutoTokenizer
import torch
from tools.read_datasets import *
from tools.utils import save_model, set_seed, extract_last_num
import argparse
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import os
from evaluation import *
import deepspeed
from tools.deepspeed_config import get_train_ds_config
from LayAlign import LayAlign,LayAlignConfig
from types import SimpleNamespace
def main():
    llm_path = "LLaMAX/LLaMAX2-7B-XNLI"
    mt_path = "google/mt5-xl"

    max_seq_len = 512
    max_gen_len = 512

    eval_batch_size = 4

    augmentation = False
    save_name = "MindMerger"
    task = "xnli"

    result_path_base = f'./results/{save_name}/{task}/'

    if 'mgsm' in task:
        test_sets = read_mgsms()
        task = 'math'
    elif 'msvamp' in task:
        test_sets = read_msvamp()
        task = 'math'
    elif 'csqa' in task:
        test_sets = read_x_csqa()
    else:
        test_sets = read_xnli()

    os.makedirs(result_path_base, exist_ok=True)
    tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # tokenizer_llm.pad_token = "[PAD]"
    print(json.dumps({
        'llm_path': llm_path,
        'mt_path': mt_path,
        'max_seq_len': max_seq_len,
        'max_gen_len': max_gen_len,
        'save_name': save_name,
        'result_path_base': result_path_base
    }, indent=2))
    print("cuda available: " , torch.cuda.is_available())
    train_micro_batch_size_per_gpu = 4
    train_batch_size = 4
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = 1
    # assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size=train_batch_size,
                                    train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
                                    gradient_accumulation_steps=gradient_accumulation,
                                    )


    encoder_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    language_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    #encoder_layers=[16,17,18,19,20,21,22,23]
    #language_layers=[0,1,2,3,4,5,6,7]
    encoder_aligner_config = {
        "encoder_hidden_dim": 2048,
        "language_hidden_dim": 4096,
        "num_transformer_submodules": 1,
        "num_attention_heads": 32,
        "num_encoder_layers": len(encoder_layers),
        "num_language_layers": len(language_layers),
        "encoder_layers": encoder_layers,
        "language_layers": language_layers,
        "projector_type": "weighted_linear",
        "batch": 4,
        "structure": "Linear"
    }
    encoder_aligner_config = SimpleNamespace(**encoder_aligner_config)

    model_config = LayAlignConfig(
        mt_path=mt_path,
        llm_path=llm_path,
        max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        encoder_aligner_config=encoder_aligner_config,
        augmentation = augmentation
    )
    init_checkpoint = "/root/LayAlign/outputs/LayAlign-xnli-test1/epoch_2_augmentation/pytorch_model.bin"
    model = LayAlign(model_config)
    if init_checkpoint is not None:
        init_checkpoint = init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        #model_dict = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint, True)
        print('mapping init from:', init_checkpoint)
    # model.to('cuda')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None)
    scores_map = {}
    avg = 0
    for test_lang in test_sets:
        test_set = test_sets[test_lang]
        test_sampler = SequentialSampler(test_set)
        test_set = MathDataset(test_set, task)
        test_set = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        if 'math' in task:
            acc, results_list = evaluate_math(model, test_set, tokenizer_llm, tokenizer_m2m,
                                                     max_seq_len, max_gen_len, augmentation, langs_map)
        else:
            acc, results_list = evaluate_classification(model, test_set, tokenizer_llm, tokenizer_m2m,
                                              max_seq_len, max_gen_len, augmentation, langs_map)
        print('test_lang:', test_lang, 'acc:', acc)
        scores_map[test_lang] = acc
        result_path = f'{result_path_base}/{test_lang}.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        avg += acc
    print(scores_map)
    print('Average accuracy :', round(avg / len(test_sets), 1))
    score_path = f'{result_path_base}/scores.tsv'
    with open(score_path, 'w', encoding='utf-8') as f:
        for lang in scores_map:
            score = scores_map[lang]
            f.write(f'{lang}\t{score}\n')



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--llm_path",
    #     type=str,
    #     default='../LLMs/MetaMath-7B-V1.0/'
    # )
    # parser.add_argument(
    #     "--mt_path",
    #     type=str,
    #     default='../LLMs/mt5-xl/'
    # )
    # parser.add_argument(
    #     "--init_checkpoint",
    #     type=str,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--save_name",
    #     type=str,
    #     default='MindMerger',
    # )
    # parser.add_argument(
    #     "--task",
    #     type=str,
    #     default='math',
    # )
    # parser.add_argument(
    #     "--eval_batch_size",
    #     type=int,
    #     default=8
    # )
    # parser.add_argument(
    #     "--local_rank",
    #     type=int,
    #     default=0
    # )
    # parser.add_argument(
    #     "--max_seq_len",
    #     type=int,
    #     default=512
    # )
    # parser.add_argument(
    #     "--max_gen_len",
    #     type=int,
    #     default=512
    # )
    # parser.add_argument(
    #     "--gpu",
    #     type=str,
    #     default='1'
    # )
    # parser.add_argument(
    #     "--augmentation",
    #     type=ast.literal_eval,
    #     default=True
    # )
    # parser.add_argument(
    #     "--train_batch_size",
    #     type=int,
    #     default=128
    # )
    # parser.add_argument(
    #     "--structure",
    #     type=str,
    #     default='Linear'
    # )
    # parser.add_argument(
    #     "--train_micro_batch_size_per_gpu",
    #     type=int,
    #     default=1
    # )
    # parser = deepspeed.add_config_arguments(parser)
    # args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    langs = ['Thai', 'Swahili', 'Bengali', 'Chinese', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'English']
    langs_map_flores = {'Swahili': 'swh', 'Bengali': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
                        'German': 'deu', 'Spanish': 'spa', 'French': 'fra', 'Japanese': 'jpn', 'Russian': 'rus', }
    langs_map_m2m = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
                     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
                     'Russian': 'ru', 'Thai': 'th', 'Greek': 'el', 'Telugu': 'te',
                     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
                     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
                     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
                     'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur'}
    langs_map_nllb = {
        'English': 'eng_Latn', 'Swahili': 'swh_Latn', 'Chinese': 'zho_Hans', 'Bengali': 'ben_Beng',
        'German': 'deu_Latn', 'Spanish': 'spa_Latn', 'French': 'fra_Latn', 'Japanese': 'jpn_Jpan',
        'Russian': 'rus_Cyrl', 'Thai': 'tha_Thai'
    }
    # if 'nllb' in args.mt_path:
    #     langs_map = langs_map_nllb
    # else:
    langs_map = langs_map_m2m
    main()