import os
import json
import math
import time
import random
import wandb
from tqdm import tqdm
from typing import Optional
import torch.autograd as autograd
import torch
torch.backends.cuda.enable_flash_sdp(False)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from discriminator import Discriminator,MLPEmbeddingLLMDiscriminator,WGANLLMDiscriminator, SimpleMLPDiscriminator

import deepspeed
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    GenerationConfig,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from accelerate import init_empty_weights
from peft import PeftModel

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from utils import (
    get_optimizer_params, get_optimizer_params_peft, print_args, initialize,
    print_rank, get_rank, save_rank, all_gather, load_parallel, save_parallel,
    get_tokenizer, get_model)
from metrics import compute_metrics
import mpu

torch.set_num_threads(4)

def get_teacher_model(args, device: int):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        config.is_model_parallel = True
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config).to(eval(args.dtype))
        load_parallel(model, args.teacher_model_path)
        model = model.to(device)
    else:
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path, 
            config=config, 
            device_map={"": device}, 
            torch_dtype=eval(args.dtype)
        )

        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model

def get_discriminator_model(args,device: int, teacher_model=None):
    if args.discrim == "Linear":
        vocab_size = teacher_model.config.vocab_size
        discriminator = Discriminator(
            vocab_size=vocab_size,
            hidden_size=768,
            num_layers=2
        )
        discriminator = discriminator.to(device).to(torch.bfloat16)
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=0.0002, weight_decay=args.weight_decay)

    elif args.discrim == "SimpleMLPDiscriminator":
        vocab_size = teacher_model.config.vocab_size
        discriminator = SimpleMLPDiscriminator(vocab_size)
        discriminator = discriminator.to(device).to(torch.bfloat16)
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=0.0002, weight_decay=args.weight_decay)
        
    elif args.discrim == "MLPEmbeddingLLMDiscriminator":
        discriminator = MLPEmbeddingLLMDiscriminator(args.discrim_path)
        discriminator = discriminator.to(device).to(torch.bfloat16)
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=0.0002, weight_decay=args.weight_decay)
    
    elif args.discrim == "WGANLLMDiscriminator":
        discriminator = WGANLLMDiscriminator(args.discrim_path)
        discriminator = discriminator.to(device).to(torch.bfloat16)
        discriminator_optimizer = RMSprop(discriminator.parameters(), lr=0.0002, weight_decay=args.weight_decay, momentum=0.9, alpha=0.99)
        
    else:
        discriminator = WGANLLMDiscriminator(args.discrim_path)
        discriminator = discriminator.to(device).to(torch.bfloat16)
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=0.0002, weight_decay=args.weight_decay)
    return discriminator, discriminator_optimizer


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
    return data


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
        distil_losses = distil_losses.view(-1)
        loss_mask = no_model_batch["loss_mask"].view(-1)
        distil_loss = (distil_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)  # p(y)
        student_probs = F.softmax(logits,         dim=-1, dtype=torch.float32)  # q(y)

        eps = 1e-9
        ratio = student_probs / (teacher_probs + eps)               # q/p
        log_ratio = torch.log(student_probs + eps) - torch.log(teacher_probs + eps)  # ln(q) - ln(p)
        term = student_probs * log_ratio
        # print("ratio",ratio)
        inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
        term = torch.masked_fill(term, inf_mask, 0.0)

        prod_probs =  term 
        
        x = torch.sum(prod_probs, dim=-1)  # [batch, seq]
        x = x.view(-1)                     # [batch*seq]

       
        mask = (no_model_batch["label"] != -100).int()
        mask = mask.view(-1)
        distil_loss = torch.sum(x * mask, dim=0) / torch.sum(mask, dim=0)
    
    return distil_loss * args.ds_scale


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device, lambda_gp=10):
    batch_size = real_samples.size(0)
    
    # construct alpha and generate interpolation samples
    alpha_shape = [batch_size] + [1] * (real_samples.dim() - 1)
    alpha = torch.rand(alpha_shape, device=device)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    interpolates.retain_grad()

    # Forward propagation gets the discriminator output
    d_interpolates = discriminator(interpolates)
    if d_interpolates.dim() > 1:
        d_interpolates = d_interpolates.view(-1)
    
    d_interpolates_sum = d_interpolates.sum()

    discriminator.zero_grad()  
    d_interpolates_sum.backward(retain_graph=True)
    
    if interpolates.grad is None:
        raise RuntimeError("Gradients cannot be calculated. Please check whether the computation graph is connected.")
    
    gradients = interpolates.grad.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    
    interpolates.grad.zero_()    
    return gradient_penalty


def finetune(args, tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer, 
             model: deepspeed.DeepSpeedEngine, optimizer: AdamW, 
             lr_scheduler, dataset, device: int,discriminator_model, discriminator_optimizer, teacher_model: Optional[PreTrainedModel] = None,
             ):
    print_rank("Start Fine-tuning")
    criterion = nn.BCELoss().to(torch.bfloat16)
    discriminator_iter = int(args.critic_it)
    generator_iter = int(args.generator_it)

    # print_inspect(model, '*')
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time, total_discriminator_loss, total_generator_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    
    evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            torch.cuda.synchronize()
            st_time = time.time()

            # -------------------------------
            # update discriminator (critic)
            # -------------------------------
            # Enable discriminator parameter updates
            for param in discriminator_model.parameters():
                param.requires_grad = True

            with torch.no_grad():
                teacher_model.eval()
                teacher_outputs = teacher_model(**model_batch, use_cache=False)
                # real sample
                teacher_logits = teacher_outputs.logits

                student_outputs = model(**model_batch, use_cache=False)
                # fake sample
                student_logits = student_outputs.logits

            batch_size = teacher_logits.shape[0]

            if epoch > -1:
                for _ in range(discriminator_iter):
                    d_real = discriminator_model(teacher_logits)
                    d_fake = discriminator_model(student_logits)

                    # WGAN-GP critic loss 
                    # higher scores for real samples and lower scores for fake samples
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                    
                    teacher_logits_gp = teacher_logits.detach().requires_grad_(True)
                    student_logits_gp = student_logits.detach().requires_grad_(True)
                    gp = compute_gradient_penalty(discriminator_model, teacher_logits_gp, student_logits_gp, device, lambda_gp=10)

                    discriminator_loss = d_loss + gp
                    discriminator_loss = discriminator_loss * args.critic_scale
                    
                    discriminator_optimizer.zero_grad()
                    
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
            else:
                discriminator_loss = 0

            # -------------------------------
            # update generator (student model)
            # -------------------------------
            for _ in range(generator_iter):
                outputs = model(**model_batch, use_cache=False)
                logits = outputs.logits

                if args.model_parallel:
                    lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"]).view(-1)
                    loss_mask = no_model_batch["loss_mask"].view(-1)
                    lm_loss = (lm_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
                else:
                    lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
                
                if teacher_model is not None:
                    distil_loss = get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits)
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
                else:
                    loss = lm_loss

                # Freeze discriminator parameters
                for param in discriminator_model.parameters():
                    param.requires_grad = False

                # generator loss： - E[critic(fake sample)]
                generator_loss = -torch.mean(discriminator_model(logits))
                if (epoch > 0):
                    loss += args.generator_scale * generator_loss

                model.backward(loss)
                model.step()

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            global_discriminator_loss = 0
            global_generator_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                
                if discriminator_loss==0:
                    global_discriminator_loss = 0
                else:   
                    dist.all_reduce(discriminator_loss, dist.ReduceOp.SUM, group=dp_group)
                    global_discriminator_loss = discriminator_loss.item() / dp_world_size

                dist.all_reduce(generator_loss, dist.ReduceOp.SUM, group=dp_group)
                global_generator_loss = generator_loss.item() / dp_world_size
                
                total_distil_loss += global_distil_loss
                total_discriminator_loss += global_discriminator_loss
                total_generator_loss += global_generator_loss
            
            
            torch.cuda.synchronize()

            log_dict = {
                "loss": global_loss,
                "distil_loss": global_distil_loss,
                "discriminator_loss": global_discriminator_loss,
                "generator_loss": global_generator_loss
            }
            # wandb.log(log_dict)
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_discriminator_loss, log_generator_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | discriminator_loss {:.4f} | generator_loss {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    log_discriminator_loss,
                    log_generator_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss,global_discriminator_loss,global_generator_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_discriminator_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_generator_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_discriminator_loss, total_generator_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        config_dict = model.module.config.to_dict()
                        if "is_model_parallel" in config_dict:
                            del config_dict["is_model_parallel"]
                        with open(os.path.join(save_dir_path, "config.json"), "w") as f:
                            json.dump(config_dict, f, indent=2)
                        tokenizer.save_pretrained(save_dir_path)
                    if mpu.get_data_parallel_rank() == 0:
                        save_parallel(model.module, save_dir_path)
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
            
    return model


def evaluate(args, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, model: PreTrainedModel, 
             dataset: LMTrainDataset, split: str, epoch: int, device: int):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            # dist.barrier()
            # for rank in range(dist.get_world_size()):
            #     if dist.get_rank() == rank:
            #         print(f"rank: {dist.get_rank()}", model_batch["input_ids"][0][:128])
            #     dist.barrier()
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"]).view(-1)
                loss_mask = no_model_batch["loss_mask"].view(-1)
                loss = (lm_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)

            log_dict = {
                "exact_match" : res["exact_match"],
                "rougeL" : res["rougeL"]
            }

            # wandb.log(log_dict)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    if "fp16" in ds_config and ds_config["fp16"]["enabled"]:
        args.dtype = "torch.float16"
    elif "bf16" in ds_config and ds_config["bf16"]["enabled"]:
        args.dtype = "torch.bfloat16"
    else:
        args.dtype = "torch.float32"
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)

 
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None

    discriminator_model, discriminator_optimizer = get_discriminator_model(args,device, teacher_model)
    
    if args.do_train:
        # wandb.init(
        #     project=args.save.split("/")[-1],
        #     entity="gan_distillation",
        #     config={
        #         "type": args.type,
        #         "learning_rate": args.lr,
        #         "batch_size": args.batch_size,
        #         "epochs": args.epochs,
        #         "kd_ratio": args.kd_ratio,
        #         "generator_scale": args.generator_scale,
        #         "critic_it": args.critic_it
        #     }
        # )
        # wandb.watch(model, log="all", log_freq=50)
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, discriminator_model,discriminator_optimizer,teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()
