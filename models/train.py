import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import os
import json
from contextlib import nullcontext
import argparse
from typing import List, Dict, Any
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from transformers import BaseImageProcessor
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import annotation.utils (which imports decord) after torch to avoid bug
import torch.multiprocessing as mp
from resource_logging import measure_resource_usage, MeasureResourceUsage
import evaluate

from models.hf_arguments import *
from backbones.language_models.cambrian_llama import CambrianLlamaForCausalLM, CambrianLlamaForSequenceClassification

from backbones import conversation as conversation_lib
from supervised_dataset import make_supervised_data_module
from grouped_sampler import LengthGroupedSampler
from opti import get_optimizer
from train_log import *
from models.utils import count_parameters, log_rank0, seed_worker, gen_hex
from models.train_ckpt import *
from models.eval import *

from backbones.mm_datautils import (
    KeywordsStoppingCriteria,
)

from collections import defaultdict
import logging
from multiprocessing import cpu_count
from backbones.constants import *

# Configure logging with line numbers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

GLOBAL_SEED = 1337
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)

ddp = int(os.environ.get("RANK", -1)) != -1

def forward_step(
    model, 
    batch, 
    device,
    model_args,  
    tokenizer,
    eval_mode=False, 
    gen_config_dict: Dict[str, Any]=None,
):  
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    position_ids = batch["position_ids"].to(device)
    image_aux_attention_masks_list = [image_aux_attn_mask.to(device) for image_aux_attn_mask in batch["image_aux_attention_masks_list"]]
    
    image_sizes = batch["image_sizes"]
    images = None
    if "images" in batch:
        assert isinstance(batch["images"], list), "images must be a list for vision tower aux"
        if isinstance(batch["images"][0], list):
            images = [[img.to(device) for img in imgs] for imgs in batch["images"]]
        else:
            images = [image.to(device) for image in batch["images"]]
    labels = labels if not eval_mode else None
    outputs = {}
    outputs["model_outputs"] = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        images=images,
        image_aux_attention_masks_list=image_aux_attention_masks_list,
        image_sizes=image_sizes,
    )
    
    conv = conversation_lib.conv_templates[model_args.version].copy()
    stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    if gen_config_dict is not None:
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        raw = model.module if hasattr(model, "module") else model
        with torch.inference_mode():
            was_training = raw.training
            if was_training:
                raw.eval() # disable dropout and checkpointing (use_cache can be True)
            output_ids = raw.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=image_sizes,
                top_p=None if not gen_config_dict.get("do_sample", False) else 0.9,
                do_sample=gen_config_dict.get("do_sample", False),
                temperature=gen_config_dict.get("temperature", 1.),
                max_new_tokens=gen_config_dict.get("max_new_tokens", 128),
                num_beams=gen_config_dict.get("num_beams", 3),
                use_cache=gen_config_dict.get("use_cache", True),
                stopping_criteria=[stopping_criteria],
            )
            if was_training:
                raw.train() # restore training mode
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # eliminate starting "assistant" prefix if present
        if pred.startswith("assistant"):
            pred = pred[len("assistant"):].strip()
        outputs["preds"] = [pred]
    return outputs

def train():
    
    if ddp:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # main process for logging, checkpointing, etc.
    else:
        # non-ddp
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        logging.info(f"Using device: {device}")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dist.barrier()

    if training_args.bf16 and training_args.fp16:
        raise ValueError("Cannot use both bf16 and fp16")

    # pyre-fixme[16]: `DataClass` has no attribute `output_model_local_path`.
    training_args.output_dir = model_args.output_model_filename
    # pyre-fixme[16]: `DataClass` has no attribute `local_dir`.
    model_args.local_dir = model_args.output_model_filename
    # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute `from_pretrained`.
    
    model = CambrianLlamaForCausalLM.from_pretrained(
        # pyre-fixme[16]: `DataClass` has no attribute `input_model_local_path`.
        model_args.input_model_filename,
    )
    model.config.use_cache = False
    # pyre-fixme[16]: `DataClass` has no attribute `freeze_backbone`.
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    # pyre-fixme[16]: `DataClass` has no attribute `gradient_checkpointing`.
    if training_args.gradient_checkpointing:
        # @tcm: might look here: https://junbuml.ee/grad-flow-lora-grad-ckpt
        model.config.use_cache = False # Disable KV-cache (mandatory with ckpt)
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.input_model_filename,
        # pyre-fixme[16]: `DataClass` has no attribute `model_max_length`.
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.pad_token_id = 128002
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        model_args.version
    ]
    log_rank0(f"Using conversation format: {conversation_lib.default_conversation.version}")
    # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_list`.
    if model_args.vision_tower_aux_list is not None:
        # pyre-fixme[16]: `DataClass` has no attribute `unfreeze_mm_vision_tower`.
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_token_len_list`.
        model_args.vision_tower_aux_token_len_list = json.loads(
            model_args.vision_tower_aux_token_len_list
        )
        # pyre-fixme[16]: `DataClass` has no attribute `query_num_list`.
        model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=None,  # FSDP or not, flag should be the same as None to avoid creation error
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()

        if not training_args.unfreeze_mm_vision_tower:
            # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(
                        dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device  # pyre-fixme
                    )
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [  # pyre-fixme
                vision_tower_aux.image_processor
                for vision_tower_aux in vision_tower_aux_list
            ]
        data_args.is_multimodal = True  # pyre-fixme

        model.config.image_aspect_ratio = data_args.image_aspect_ratio  # pyre-fixme
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position  # pyre-fixme
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token  # pyre-fixme

        # pyre-fixme
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            tune_modules = [
                "mm_projector",
                "pos_emb",
                "vision_sampler",
                "vision_sampler_layers",
                "vision_query",
                "image_newline",
            ]
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    param.requires_grad = True
        if model_args.tune_lm_head:
            for p in model.lm_head.parameters():
                p.requires_grad = True
        if model_args.tune_embed_tokens:
            for p in model.get_input_embeddings().parameters():
                p.requires_grad = True
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter  # pyre-fixme
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.image_token_len = model_args.image_token_len = (  # pyre-fixme
            model_args.image_token_len
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr  # pyre-fixme
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr  # pyre-fixme
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr  # pyre-fixme
        training_args.use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = (
            data_args.vision_tower_aux_token_len_list
        ) = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = model_args.image_token_len
        model.config.is_st_sampler = model_args.is_st_sampler  # pyre-fixme
        data_args.image_token_len = model_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        log_rank0(f"After initializing tokenizer, vocab size: {len(tokenizer)}")
    
    if training_args.bf16:
        model.to(torch.bfloat16)
    elif training_args.fp16:
        model.to(torch.float16)
    model.to(device)
    # pyre-fixme
    def convert_bn_to_float(model):
        if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
            return model.float()
        for child_name, child in model.named_children():
            model.add_module(child_name, convert_bn_to_float(child))
        return model

    model = convert_bn_to_float(model)
    if master_process:
        count_parameters(model, print_layers = True)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    log_rank0("Wrapped in DDP")

    num_epochs: int = training_args.num_train_epochs
    last_epoch: int = 0
    batch_in_last_epoch: int = -1
    global_steps: int = 0
    gradient_accumulation_steps: int = int(training_args.gradient_accumulation_steps)
    # Create a reproducible generator
    generator = torch.Generator()
    epoch_seed: int = GLOBAL_SEED
    if training_args.resume_from_checkpoint is not None:
        checkpoint_base_dir = os.path.dirname(training_args.resume_from_checkpoint)
        assert os.path.isdir(checkpoint_base_dir), f"Checkpoint base dir {checkpoint_base_dir} does not exist"
        assert os.path.isfile(training_args.resume_from_checkpoint), f"Checkpoint {training_args.resume_from_checkpoint} is not a file"
        ckpt = load_checkpoint(
            training_args.resume_from_checkpoint, 
            training_args,
            model,
            generator, # load rng states
            load_optimizer=True, 
            load_scheduler=True
        )
        log_rank0("Loaded checkpoint")
        world_size = ckpt["world_size"]
        assert world_size == ddp_world_size, f"World size mismatch: ckpt world size = {world_size} != current world size = {ddp_world_size}"
        ckpt_gradient_accumulation_steps = ckpt["gradient_accumulation_steps"]
        assert gradient_accumulation_steps == ckpt_gradient_accumulation_steps, f"Gradient accumulation steps mismatch: ckpt gradient accumulation steps = {ckpt_gradient_accumulation_steps} != current gradient accumulation steps = {gradient_accumulation_steps}"
        ckpt_per_device_train_batch_size = ckpt["per_device_train_batch_size"]
        assert training_args.per_device_train_batch_size == ckpt_per_device_train_batch_size, f"Per-device train batch size mismatch: ckpt per-device train batch size = {ckpt_per_device_train_batch_size} != current per-device train batch size = {training_args.per_device_train_batch_size}"
        ckpt_rank = ckpt["rank"]
        assert ckpt_rank == ddp_rank, f"Rank mismatch: ckpt rank = {ckpt_rank} != current rank = {ddp_rank}"

        optimizer = ckpt["optimizer"]
        scheduler = ckpt["scheduler"]
        last_epoch = ckpt["last_epoch"]
        batch_in_last_epoch = ckpt["batch_in_last_epoch"]
        global_steps = ckpt["global_steps"]
        num_warmup_steps = ckpt["num_warmup_steps"]
        num_training_steps = ckpt["num_training_steps"]
        epoch_seed = ckpt["epoch_seed"]
    else:
        generator.manual_seed(GLOBAL_SEED)
    # log_rank0(f"[DBG] after _load / manual_seed  : gen={gen_hex(generator)}")
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    eval_dataset = data_module["eval_dataset"]
    data_collator = data_module["data_collator"]

    assert training_args.group_by_modality_length is True, "Group by modality length must be True"
    # Instantiate LengthGroupedSampler
    train_sampler = LengthGroupedSampler(
        batch_size=training_args.per_device_train_batch_size,
        world_size=ddp_world_size,
        lengths=train_dataset.modality_lengths,
        generator=generator,
        group_by_modality=training_args.group_by_modality_length,
    )
    # log_rank0(f"[DBG] just built sampler: gen={gen_hex(generator)}")
    eval_sampler = LengthGroupedSampler(
        batch_size=training_args.per_device_eval_batch_size,
        world_size=ddp_world_size,
        lengths=eval_dataset.modality_lengths,
        generator=generator,
        group_by_modality=training_args.group_by_modality_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True, # per-device train batch size = 1 so we won't miss too many samples
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True, # per-device eval batch size = 1 so we won't miss too many samples
    )
    if training_args.resume_from_checkpoint is None:
        optimizer = get_optimizer(model, training_args)
        num_training_steps = (len(train_dataloader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps * num_epochs
        num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)  # warm up % of training steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    from_epoch = last_epoch
    from_batch = batch_in_last_epoch + 1
    if batch_in_last_epoch >= len(train_dataloader):
        from_epoch += 1
        from_batch = 0
    
    logging_steps: int = int(training_args.logging_steps)
    eval_steps: int = int(training_args.eval_steps)
    save_steps: int = int(training_args.save_steps)
    if master_process:
        if os.path.exists(training_args.output_dir):
            raise ValueError(f"Output directory {training_args.output_dir} already exists")
        os.makedirs(training_args.output_dir, exist_ok=True)
    train_log_fpath = os.path.join(training_args.output_dir, training_args.train_log)
    train_perf_log_fpath = os.path.join(training_args.output_dir, training_args.train_perf_log)
    eval_perf_log_fpath = os.path.join(training_args.output_dir, training_args.eval_perf_log)
    eval_log_fpath = os.path.join(training_args.output_dir, training_args.eval_log)
    train_logs: List[TrainProgressLog] = []
    train_perf: List[PerfMetrics] = []
    eval_perf: List[PerfMetrics] = []
    if training_args.generation_eval:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")

    log_rank0("Starting training")
    for epoch in range(from_epoch, num_epochs):
        if ddp:
            # Ensure each process sees a different ordering at each epoch
            if epoch > last_epoch:
                epoch_seed = GLOBAL_SEED + epoch
                generator.manual_seed(epoch_seed)
        model.train()
        train_loss_accum = torch.zeros(1, device=device)
        train_device_preds, train_device_gold_labels = [], []
        train_device_text_preds, train_device_text_references = [], []
        for batch_idx, batch in enumerate(train_dataloader):
            # if epoch == from_epoch and batch_idx == from_batch:
            #     log_rank0(f"[DBG] first batch this run: gen={gen_hex(generator)}")
            if epoch == from_epoch and batch_idx < from_batch:
                log_rank0(f"Skipping epoch {epoch} batch {batch_idx}")
                continue
            log_rank0(f'Epoch {epoch + 1}/{num_epochs}, batch {batch_idx + 1}/{len(train_dataloader)}')

            is_last_micro = ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx == len(train_dataloader) - 1)
            
            # DDP: skip gradient synchronisation on all but final micro-step
            ddp_context = model.no_sync() if (ddp and not is_last_micro) else nullcontext()
            train_labels = batch["eng_classes"].to(device)
            batch_eng = {
                "input_ids": batch["input_ids_eng"],
                "attention_mask": batch["attention_mask_eng"],
                "position_ids": batch["position_ids_eng"],
                "labels": batch["labels_eng"],
                "images": batch["images"],
                "image_aux_attention_masks_list": batch["image_aux_attention_masks_list_eng"],
                "image_sizes": batch["image_sizes"],
            }
            batch_rationale = {
                "input_ids": batch["input_ids_rationale"],
                "attention_mask": batch["attention_mask_rationale"],
                "position_ids": batch["position_ids_rationale"],
                "labels": batch["labels_rationale"],
                "images": batch["images"],
                "image_aux_attention_masks_list": batch["image_aux_attention_masks_list_rationale"],
                "image_sizes": batch["image_sizes"],
            }
            with ddp_context:
                outputs_eng = forward_step(
                    model, 
                    batch_eng, 
                    device, 
                    model_args, 
                    tokenizer,
                    gen_config_dict={
                        "do_sample": False,
                        "max_new_tokens": 256,
                        "num_beams": 1,
                        "use_cache": True,
                    } if training_args.generation_eval else None
                )
                # loss = 0.5 * outputs["engagement"].loss + 0.5 * outputs["rationale"].loss # I predict the CUDA OOM error stems from here, where loss graphs of two forward passes are combined
                lbd = training_args.cls_loss_weight
                loss_eng = outputs_eng["model_outputs"].loss
                loss_eng = lbd * loss_eng / (2. * gradient_accumulation_steps)
                train_loss_accum += loss_eng.detach()
                loss_eng.backward()
                
                outputs_rationale = forward_step(
                    model, 
                    batch_rationale, 
                    device, 
                    model_args, 
                    tokenizer,
                    gen_config_dict={
                        "do_sample": False,
                        "max_new_tokens": 256,
                        "num_beams": 1,
                        "use_cache": True,
                    } if training_args.generation_eval else None
                )
                loss_rationale = outputs_rationale["model_outputs"].loss
                loss_rationale = (1. - lbd) * loss_rationale / (2. * gradient_accumulation_steps)
                train_loss_accum += loss_rationale.detach()
                loss_rationale.backward()
            
            if is_last_micro:
                # Update weights every accum_steps mini-batches
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                global_steps += 1
                if ddp:
                    dist.all_reduce(train_loss_accum, op=dist.ReduceOp.AVG)
                if global_steps % logging_steps == 0 and master_process:
                    total_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    logging.info(f'Epoch {epoch + 1}/{num_epochs}, global step: {global_steps}, loss={train_loss_accum.item():.10f}, clipped gradient norm: {total_norm:.4f}')
                    train_logs.append(TrainProgressLog(
                        run_type="train",
                        epoch=epoch + (batch_idx+1) / len(train_dataloader),
                        step=global_steps,
                        loss=train_loss_accum.item(),
                        grad_norm=total_norm,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        # @tcm: At the moment, stop printing out predicted label and text for last video in the batch at logging steps because of longer training
                        # video_path=batch["video_paths"][0],
                        # cls_pred=cur_preds.item(),
                        # gen_pred=outputs["preds"][0] if training_args.generation_eval else None
                    ))
                    with open(train_log_fpath, "w") as f:
                        json_train_logs = [log.to_dict() for log in train_logs]
                        json.dump(json_train_logs, f, indent=4)
                    # for param_group in optimizer.param_groups:
                    #     cur_lr = param_group["lr"]
                    #     logging.info(f'lr: {cur_lr:.10f}')
                
                # zero out grads for all original tokens, keep <cls> trainable
                with torch.no_grad():
                    grad = model.module.get_input_embeddings().weight.grad if hasattr(model, "module") else model.get_input_embeddings().weight.grad
                    if grad is not None:
                        assert grad.ndim == 2, "require grad.ndim == 2"
                        grad[:-1, :] = 0          # zero out grads for all original tokens
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)              # clear for next cycle
                train_loss_accum.zero_()        # reset tensor, keeps same device

                do_eval = False
                if training_args.eval_strategy == 'epoch':
                    do_eval = (batch_idx == len(train_dataloader) - 1)
                elif training_args.eval_strategy == 'steps':
                    do_eval = (global_steps % eval_steps == 0)
                # If `"epoch"` or `"steps"` is chosen, saving will also be performed at the very end of training, always.
                if epoch == num_epochs - 1 and batch_idx == len(train_dataloader) - 1:
                    do_eval = True
                if do_eval:
                    # evaluate on the training fraction first
                    # train_perf_log = evaluate_perf(
                    #     device_preds=train_device_preds,
                    #     device_gold_labels=train_device_gold_labels,
                    #     prefix="Train",
                    #     predictions=None,
                    #     references=None,
                    #     epoch=epoch + (batch_idx+1) / len(train_dataloader),
                    #     step=global_steps,
                    # )
                    # if train_perf_log is not None:
                    #     # only on master process
                    #     train_perf.append(train_perf_log)
                    #     with open(train_perf_log_fpath, "w") as f:
                    #         json_train_perf = [perf.to_dict() for perf in train_perf]
                    #         json.dump(json_train_perf, f, indent=4)

                    if ddp:
                        dist.barrier() # wait for all processes to finish before evaluation
                    model.eval()
                    
                    eval_device_loss = 0.
                    eval_device_samples = 0
                    # eval_device_preds, eval_device_gold_labels = [], []
                    # eval_device_text_preds, eval_device_text_references = [], []
                    # eval_video_paths = []

                    eval_engagement_preds = []
                    for eval_batch_idx, eval_batch in enumerate(eval_dataloader):
                        log_rank0(f'After epoch {epoch + 1}, eval batch {eval_batch_idx+1}/{len(eval_dataloader)}')

                        eval_label = int(eval_batch["eng_classes"][0])
                        eval_batch_eng = {
                            "input_ids": eval_batch["input_ids_eng"],
                            "attention_mask": eval_batch["attention_mask_eng"],
                            "position_ids": eval_batch["position_ids_eng"],
                            "labels": eval_batch["labels_eng"],
                            "images": eval_batch["images"],
                            "image_aux_attention_masks_list": eval_batch["image_aux_attention_masks_list_eng"],
                            "image_sizes": eval_batch["image_sizes"],
                        }
                        with torch.no_grad():
                            outputs = forward_step(
                                model, 
                                eval_batch_eng, 
                                device, 
                                model_args, 
                                tokenizer, 
                                eval_mode=True, 
                                gen_config_dict={
                                    "do_sample": False,
                                    "max_new_tokens": 16,
                                    "num_beams": 1,
                                    "use_cache": True,
                                },
                            )
                            eval_engagement_preds.append({
                                "video_path": eval_batch["video_paths"][0],
                                "engagement_pred": outputs["preds"][0],
                                "gold_label": eval_label
                            })
                            # eval_logits = outputs.cls_logits
                            # cur_preds = torch.argmax(eval_logits, dim=-1)
                            # eval_device_preds.append(cur_preds)
                            # eval_device_gold_labels.append(eval_labels)
                            # loss_fnc = torch.nn.CrossEntropyLoss()
                            # eval_loss = loss_fnc(eval_logits, eval_labels)
                            # eval_device_loss += eval_loss.item() * eval_labels.shape[0]
                            # eval_device_samples += eval_labels.shape[0]
                            # if training_args.generation_eval:
                            #     eval_device_text_preds.extend(outputs["preds"])
                            #     eval_device_text_references.extend(eval_batch["responses"])
                            # eval_video_paths.extend(eval_batch["video_paths"])

                    all_engagement_preds = [None for _ in range(ddp_world_size)] if master_process else None
                    dist.gather_object(eval_engagement_preds, all_engagement_preds, dst=0)
                    if master_process:
                        cur_eval_log_fname = os.path.basename(eval_log_fpath).split(".")[0] + f"-epoch{epoch}-step{global_steps}.json"
                        cur_eval_log_fdir = os.path.dirname(eval_log_fpath)
                        cur_eval_log_fpath = os.path.join(cur_eval_log_fdir, cur_eval_log_fname)
                        with open(cur_eval_log_fpath, "w") as f:
                            json.dump(all_engagement_preds, f, indent=4)

                    # if master_process and training_args.generation_eval:
                    #     # logging.info(f"Eval video paths: {eval_video_paths}")
                    #     # @tcm: At the moment, print out predicted label and generated text for each video in the eval set.
                    #     assert len(eval_video_paths) == len(eval_device_text_preds) and len(eval_video_paths) == len(eval_device_preds), "need equal"
                    #     eval_logs: List[EvalProgressLog] = []
                    #     for video_path, cls_pred, gen_pred in zip(eval_video_paths, eval_device_preds, eval_device_text_preds):
                    #         eval_logs.append(EvalProgressLog(
                    #             epoch=epoch + (batch_idx+1) / len(train_dataloader),
                    #             step=global_steps,
                    #             video_path=video_path,
                    #             cls_pred=cls_pred.item(),
                    #             gen_pred=gen_pred
                    #         ))
                    #     cur_eval_log_fname = os.path.basename(eval_log_fpath).split(".")[0] + f"-epoch{epoch}-step{global_steps}.json"
                    #     cur_eval_log_fdir = os.path.dirname(eval_log_fpath)
                    #     cur_eval_log_fpath = os.path.join(cur_eval_log_fdir, cur_eval_log_fname)
                    #     with open(cur_eval_log_fpath, "w") as f:
                    #         json_eval_logs = [log.to_dict() for log in eval_logs]
                    #         json.dump(json_eval_logs, f, indent=4)

                    # @tcm: Open this when you need evaluate text matching metrics (bleu, rouge, bertscore, etc.)
                    # eval_gathered_preds = [None for _ in range(ddp_world_size)] if master_process else None
                    # eval_gathered_references = [None for _ in range(ddp_world_size)] if master_process else None
                    # if training_args.generation_eval:
                    #     dist.gather_object(eval_device_text_preds, eval_gathered_preds, dst=0)
                    #     dist.gather_object(eval_device_text_references, eval_gathered_references, dst=0)
                    # if master_process and training_args.generation_eval:
                    #     # flatten
                    #     eval_gathered_preds = [pred for rank_preds in eval_gathered_preds for pred in rank_preds]
                    #     eval_gathered_references = [ref for rank_refs in eval_gathered_references for ref in rank_refs]
                    # text_evaluators = {}
                    # if training_args.generation_eval:
                    #     text_evaluators = {"bleu": bleu, "rouge": rouge, "meteor": meteor, "bertscore": bertscore}
                    # eval_perf_log = evaluate_perf(
                    #     device_loss=eval_device_loss,
                    #     device_samples=eval_device_samples,
                    #     device_preds=eval_device_preds,
                    #     device_gold_labels=eval_device_gold_labels,
                    #     predictions=eval_gathered_preds if training_args.generation_eval else None,
                    #     references=eval_gathered_references if training_args.generation_eval else None,
                    #     prefix="Eval",
                    #     epoch=epoch + (batch_idx+1) / len(train_dataloader),
                    #     step=global_steps,
                    #     **text_evaluators
                    # )
                    # if eval_perf_log is not None:
                    #     # only on master process
                    #     eval_perf.append(eval_perf_log)
                    #     with open(eval_perf_log_fpath, "w") as f:
                    #         json_eval_perf = [perf.to_dict() for perf in eval_perf]
                    #         json.dump(json_eval_perf, f, indent=4)
                    model.train()
    
                do_save = False
                checkpoint_name = model_args.checkpoint_fname
                if training_args.save_strategy == 'epoch':
                    do_save = (batch_idx == len(train_dataloader) - 1)
                    if do_save:
                        checkpoint_name = f'{checkpoint_name}-epoch{epoch}.pt'
                elif training_args.save_strategy == 'steps':
                    do_save = (global_steps % save_steps == 0)
                    if do_save:
                        checkpoint_name = f'{checkpoint_name}-epoch{epoch}-step{global_steps}.pt'
                if epoch == num_epochs - 1 and batch_idx == len(train_dataloader) - 1:
                    # always save checkpoint at the very last training step
                    do_save = True
                    checkpoint_name = f'{checkpoint_name}-epoch{epoch}-final.pt'
                if do_save:
                    log_rank0(f'Saving checkpoint at epoch {epoch}, global step {global_steps}...')
                    checkpoint_path = os.path.join(model_args.output_model_filename, checkpoint_name)
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        num_warmup_steps,
                        num_training_steps,
                        last_epoch=epoch,
                        batch_in_last_epoch=batch_idx,
                        global_steps=global_steps,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        per_device_train_batch_size=training_args.per_device_train_batch_size,
                        world_size=ddp_world_size,
                        epoch_seed=epoch_seed,
                    )
        
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    # os.makedirs(SAFETENSORS_PATH, exist_ok=True)
    # mp.set_start_method('spawn')
    train()
