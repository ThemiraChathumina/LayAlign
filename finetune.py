#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch.fx
from tqdm import tqdm
from transformers import AutoTokenizer
from tools.read_datasets import *
import argparse
import ast
from torch.utils.data import DataLoader, SequentialSampler
import json
from tools.input_features import *
from LayAlign import LayAlign,LayAlignConfig
from evaluation import accelerate_evaluate_ppl
import wandb
from dotenv import load_dotenv
from types import SimpleNamespace
from transformers import AutoTokenizer, get_scheduler
from accelerate.utils import set_seed


logger = get_logger(__name__)

import os

directory = "./datas/query_translation/"
os.makedirs(directory, exist_ok=True)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_path",
        type=str,
        default='../LLMs/Llama-2-7b-hf/'
    )
    parser.add_argument(
        "--mt_path",
        type=str,
        default='../LLMs/mt5-xl/'
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='MindMerger'
    )
    parser.add_argument(
        "--stage_name",
        type=str,
        default='translation'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=3
    )
    parser.add_argument(
        "--train_num",
        type=int,
        default=100000
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=24
    )
    parser.add_argument(
        "--train_micro_batch_size_per_gpu",
        type=int,
        default=1
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=24
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=512
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        default=3000
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--structure",
        type=str,
        default='Linear'
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--augmentation",
        type=ast.literal_eval,
        default=False
    )
    parser.add_argument(
        "--enc_freeze",
        type=ast.literal_eval,
        default=True
    )
    parser.add_argument(
        "--llm_freeze",
        type=ast.literal_eval,
        default=True
    )
    parser.add_argument(
        "--acc_cal_step",
        type=int,
        default=2000000000000,
        help="Log the ppl every acc_cal_step steps.",
    )
    parser.add_argument(
        "--warm_rate",
        type=float,
        default=0.1,
        help="warm rate.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        default=None,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None
    )
    parser.add_argument(
        "--lr_scheduler_name",
        type=str,
        default="cosine",
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    return args

def save_with_accelerate(accelerator, model, output_dir, model_name='pytorch_model.bin'):
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir
    accelerator.wait_for_everyone()
    accelerator.save_model(model, output_file, max_shard_size="30GB",safe_serialization=False)





def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        load_dotenv()
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
        os.environ["WANDB_MODE"] = "offline"

    # if you get timeouts (e.g. due to long tokenization) increase this.
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        **accelerator_log_kwargs
    )

    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    

    accelerator.wait_for_everyone()
    set_seed(0)
    langs = ['Thai', 'Swahili', 'Bengali', 'Chinese', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'English']
    langs_map_flores = {'Swahili': 'swh', 'Benli': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
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

    if 'nllb' in args.mt_path:
        langs_map = langs_map_nllb
    else:
        langs_map = langs_map_m2m
    llm_path = args.llm_path
    mt_path = args.mt_path
    train_num = args.train_num
    stage_name = args.stage_name
    task = args.task
    if stage_name == 'translation':
        if 'math' in task:
            languages = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                         'Spanish']
            train_set = read_lego(train_num, languages)

        elif 'csqa' in task:
            languages = ['Urdu', 'Hindi', 'Swahili', 'Japanese', 'Vietnamese', 'Polish', 'Chinese',
                         'Flemish', 'Russian', 'Italian', 'German', 'Portuguese', 'French', 'Spanish', 'Arabic']
            train_set = read_lego(train_num, languages)

        elif 'xnli' in task:
            languages = ['Swahili', 'Urdu', 'Hindi', 'Thai', 'Arabic', 'Turkish', 'Greek',
                          'Vietnamese', 'Chinese', 'Russian', 'Bulgarian', 'German', 'French', 'Spanish']
            train_set = read_lego(train_num, languages)
        else:
            raise ValueError("mapping no task")
        task = 'translation'
    else:
        # we don't calculate the ppl in task stage.
        if 'math' in task:
            train_set = read_math_train(train_num)
        elif 'csqa' in task:
            train_set = read_x_csqa_train()
        elif 'xnli' in task:
            train_set = read_xnli_train()
        else:
            raise ValueError("augmentation no task")

    dev_set = train_set[:args.dev_size]
    train_set = train_set[args.dev_size:]
    train_set = MathDataset(train_set, task)
    dev_set = MathDataset(dev_set, task)
    dev_set = MathDataset(dev_set, task)
    lr = args.lr
    epoch_num = args.epoch_num
    gradient_accumulation = args.gradient_accumulation
    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu

    augmentation = args.augmentation
    save_name = args.save_name
    result_path_base = f'./results/{save_name}/{stage_name}/'
    output_model_path_base = f'./outputs/{save_name}/{stage_name}/'
    tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # tokenizer_llm.pad_token = "[PAD]"

    print(json.dumps({
        'llm_path': llm_path,
        'mt_path': mt_path,
        'lr': lr,
        'epoch_num': epoch_num,
        'gradient_accumulation': gradient_accumulation,
        'train_set:': len(train_set),
        'dev_set:': len(dev_set),
        'max_seq_len': max_seq_len,
        'max_gen_len': max_gen_len,
        'train_batch_size': train_batch_size,
        'result_path': result_path_base,
        'output_model_path': output_model_path_base,
    }, indent=2))

    if stage_name != 'translation' and args.init_checkpoint is None:
        args.init_checkpoint = f'./outputs/{save_name}/translation/pytorch_model.bin'
    


    encoder_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    language_layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
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
        "batch": args.train_micro_batch_size_per_gpu,
        "structure": args.structure
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

    model = LayAlign(model_config)

    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        #model_dict = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint, True)
        print('mapping init from:', init_checkpoint)
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, requires_grad={param.requires_grad}, shape={param.shape}")
    #train_sampler = RandomSampler(train_set)
    dev_sampler = SequentialSampler(dev_set)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=train_micro_batch_size_per_gpu,
        shuffle=True
    )
    dev_dataloader = DataLoader(
        dataset=dev_set,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=1,
        drop_last=False)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_set)), 3):
        logger.info(f"Sample {index} of the training set: {train_set[index]}.")
    
    # Optimizer
    optimizer = torch.optim.AdamW(parameters, betas=[0.8,0.999], eps=1e-8, weight_decay=3e-7, lr=args.lr)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation)
    max_train_steps = args.epoch_num * num_update_steps_per_epoch
    overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parLayAlignl training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = max_train_steps if overrode_max_train_steps else max_train_steps * accelerator.num_processes
    """
    get_scheduler Agrs
    name:
        LINEAR = "linear"
        COSINE = "cosine"
        COSINE_WITH_RESTARTS = "cosine_with_restarts"
        POLYNOMIAL = "polynomial"
        CONSTANT = "constant"
        CONSTANT_WITH_WARMUP = "constant_with_warmup"
        INVERSE_SQRT = "inverse_sqrt"
        REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    """
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_name,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warm_rate),
    )


    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation)
    if overrode_max_train_steps:
        max_train_steps = epoch_num * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    epoch_num = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        accelerator.init_trackers(args.wandb_project_name, experiment_config)
        # 设置 WandB 的运行名称
        if accelerator.is_main_process:
            wandb.run.name = args.wandb_name
        #accelerator.get_tracker("wandb").run.name = args.wandb_name
    # Train!
    total_batch_size = train_micro_batch_size_per_gpu * accelerator.num_processes * gradient_accumulation

    logger.info("***** Running training *****")
    logger.info(f"  Num examples transet = {len(train_set)}")
    logger.info(f"  Num examples dataloader = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {epoch_num}")
    logger.info(f"  Instantaneous batch size per device = {train_micro_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parLayAlignl, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  parameters = {parameters}")
    logger.info(f"  optimizer = {optimizer}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    progress_bar.update(completed_steps)

    best_perplexity = 100000000000000
    
    logger.info(f"  best_perplexity = {best_perplexity}")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        total_regularization_loss = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                sources = batch['source']
                prompts = batch['prompt']
                targets = batch['target']
                source_languages = batch['source_language']

                input_ids_m2m, attention_mask_m2m = mt_input_features(sources, tokenizer_m2m,
                                                                    max_seq_len, source_languages,
                                                                    langs_map)
                add_bos_token = False
                add_eos_token = True
                labels, mask_label = llm_input_features(targets, tokenizer_llm,
                                                        max_gen_len, add_bos_token, add_eos_token)

                input_ids_prompt, mask_prompt = None, None
                if augmentation:
                    add_bos_token = False
                    add_eos_token = False
                    input_ids_prompt, mask_prompt = llm_input_features(prompts, tokenizer_llm,
                                                                        max_gen_len, add_bos_token,
                                                                        add_eos_token)
                output_loss = model(input_ids_m2m, attention_mask_m2m,
                            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
                            labels=labels, mask_label=mask_label)
                loss = output_loss
                total_loss += output_loss.detach().float()
                # We keep track of the loss at each logged step
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / gradient_accumulation / args.logging_steps
                    total_loss = 0                   
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        log_data = {
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if augmentation:
                            log_data["aug_train_loss"] = avg_loss
                        else:
                            log_data["train_loss"] = avg_loss
                        
                        accelerator.log(log_data, step=completed_steps)
                    
                if args.acc_cal_step and completed_steps % args.acc_cal_step == 0:
                    perplexity = accelerate_evaluate_ppl(accelerator, model, dev_dataloader, tokenizer_llm, tokenizer_m2m,
                                          max_seq_len, max_gen_len, langs_map, augmentation)
                    
                    if perplexity < best_perplexity:
                        best_perplexity = perplexity
                        save_with_accelerate(accelerator, model, output_model_path_base)
                        logger.info('save new best')
                    if args.with_tracking:
                        if augmentation:
                            accelerator.log(
                                {
                                    "perplexity": perplexity
                                },
                                step=completed_steps,
                            )
                        else:
                            accelerator.log(
                                {
                                    "mapping_perplexity": perplexity
                                },
                                step=completed_steps,
                            )
                if completed_steps % 1000 == 0 and completed_steps > 0 and augmentation:
                    step_model_path = f'./outputs/{save_name}/step_{completed_steps}_{stage_name}/'
                    #save_with_accelerate(accelerator, model, step_model_path)
                    print('save epoch model')
                    
        epoch_model_path = f'./outputs/{save_name}/epoch_{epoch}_{stage_name}/'
        save_with_accelerate(accelerator, model, epoch_model_path)
        print('save epoch model')
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
