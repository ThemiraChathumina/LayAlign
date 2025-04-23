import os
import random
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset
from huggingface_hub import login, upload_file
import json
import torch
import math
from model import MPTModel
from utils import set_seed, save_with_accelerate
from read_data import ExperimentDataset, llm_input_features

class Arguments:
    def __init__(self):
        BATCH_SIZE_PER_GPU=4
        TOTAL_BATCH_SIZE=8
        GRADIENT_ACC_STEPS = TOTAL_BATCH_SIZE // BATCH_SIZE_PER_GPU

        self.llm_path = "LLaMAX/LLaMAX2-7B-XNLI"
        self.mt_path = "google/mt5-xl"
        self.ext_path = "facebook/nllb-200-distilled-600M"
        self.train_num = 8888
        self.dev_size = 1000
        self.lr = 3e-5
        self.epoch_num = 3
        self.gradient_accumulation = GRADIENT_ACC_STEPS
        self.max_seq_len = 200
        self.max_gen_len = 200
        self.train_batch_size = TOTAL_BATCH_SIZE
        self.eval_batch_size = BATCH_SIZE_PER_GPU
        self.train_micro_batch_size_per_gpu = BATCH_SIZE_PER_GPU
        self.augmentation = False
        self.save_name = 'xnli'
        self.stage_name = 'no_aug'
        self.report_to = 'wandb'
        self.logging_steps = 10
        self.warm_rate = 0.05
        self.lr_scheduler_name = 'cosine'
        self.system_prompt = None
        self.init_checkpoint = None

def main():
    
    args = Arguments()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator_log_kwargs = {}
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        **accelerator_log_kwargs
    )

    accelerator.wait_for_everyone()
    set_seed(0)

    llm_path = args.llm_path
    mt_path = args.mt_path
    ext_path = args.ext_path
    
    token = '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'
    login(token=token)

    train_samples = []

    ds = load_dataset("facebook/xnli", "en")

    train_ds = ds['train']

    for i in range(100000):
        example = train_ds[i]
        premise = example['premise']
        hypothesis = example['hypothesis']
        labels = {0:'Entailment', 1:'Neutral', 2:'Contradiction'}
        sentence = f"Premise: {premise.strip()} Hypothesis: {hypothesis.strip()} Label:"
        label = f"{labels[int(example['label'])]}"
    
        sample = {
            'prompt': sentence,
            'target': label
        }

        train_samples.append(sample)

    args.train_num = len(train_samples)
    
    train_set = train_samples[args.dev_size:]
    dev_set = train_samples[:args.dev_size]
    
    dev_set = ExperimentDataset(
        dev_set
    )
    
    train_set = ExperimentDataset(
        train_set
    )

    train_num = args.train_num
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
    stage_name = args.stage_name
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
        'ext_path': ext_path,
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


    model_config = {
        'mt_path': mt_path,
        'ext_path': ext_path,
        'llm_path': llm_path,
        'max_gen_len': max_gen_len,
        'llm_bos_token_id': tokenizer_llm.bos_token_id,
        'llm_pad_token_id': tokenizer_llm.pad_token_id,
        'augmentation' :  augmentation,
        'max_seq_len': max_seq_len
    }

    model = MPTModel(model_config)

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
        print(f"Sample {index} of the training set: {train_set[index]}.")
    
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
    
    # Train!
    total_batch_size = train_micro_batch_size_per_gpu * accelerator.num_processes * gradient_accumulation

    print("***** Running training *****")
    print(f"  Num examples transet = {len(train_set)}")
    print(f"  Num examples dataloader = {len(train_dataloader)}")
    print(f"  Num Epochs = {epoch_num}")
    print(f"  Instantaneous batch size per device = {train_micro_batch_size_per_gpu}")
    print(f"  Total train batch size (w. parLayAlignl, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation}")
    print(f"  Total optimization steps = {max_train_steps}")
    print(f"  parameters = {parameters}")
    print(f"  optimizer = {optimizer}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    progress_bar.update(completed_steps)
    
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                sources = batch['prompt']
                targets = batch['target']

                add_bos_token = False
                add_eos_token = True
                labels, mask_label = llm_input_features(targets, tokenizer_llm,
                                                        max_gen_len, add_bos_token, add_eos_token)

                input_ids_prompt, mask_prompt = None, None
                if augmentation:
                    add_bos_token = False
                    add_eos_token = False
    
                    llm_input_prompts = [i for i in sources]
                    
                    # if args.system_prompt is not None:
                    #     user_prompts = [
                    #         [{
                    #             "role": "system", "content": args.system_prompt
                    #         },
                    #         {
                    #             "role": "user", "content": user_prompt_function(sources[i])
                    #         }]
                    #         for i in range(len(sources))
                    #     ]

                    #     llm_input_prompts = [tokenizer_llm.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in user_prompts]
                        
                    input_ids_prompt, mask_prompt = llm_input_features(llm_input_prompts, tokenizer_llm,
                                                                        max_gen_len, add_bos_token,
                                                                        add_eos_token)
                output_loss = model(sources,
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
                    print(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                  
        epoch_model_path = f'./outputs/{save_name}/epoch_{epoch}_{stage_name}/'
        save_with_accelerate(accelerator, model, epoch_model_path)
        print('save epoch model')
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()