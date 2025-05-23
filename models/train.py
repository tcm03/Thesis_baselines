import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR # for verifying the correctness of checkpoint resumption only, no real use

import os
import json
from contextlib import nullcontext
import argparse
from typing import List, Dict
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from transformers import BaseImageProcessor
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import annotation.utils (which imports decord) after torch to avoid bug
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
import torch.multiprocessing as mp
from resource_logging import measure_resource_usage, MeasureResourceUsage

from hf_arguments import *
from backbones.language_models.cambrian_llama import CambrianLlamaForCausalLM
from backbones import conversation as conversation_lib
from supervised_dataset import make_supervised_data_module
from grouped_sampler import LengthGroupedSampler
from opti import get_optimizer
from train_log import *
from models.utils import count_parameters, log_rank0, seed_worker
from models.train_ckpt import *

from collections import defaultdict
import logging
from multiprocessing import cpu_count
from backbones.constants import *

# Configure logging with line numbers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

GLOBAL_SEED = 1337
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)

ddp = int(os.environ.get("RANK", -1)) != -1

def forward_step(model, batch, device, eval_mode=False):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    eng_classes = batch["eng_classes"].to(device)
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
    if eval_mode:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # labels=labels,
            # eng_classes=eng_classes,
            images=images,
            image_aux_attention_masks_list=image_aux_attention_masks_list,
            image_sizes=image_sizes,
        )
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            eng_classes=eng_classes,
            images=images,
            image_aux_attention_masks_list=image_aux_attention_masks_list,
            image_sizes=image_sizes,
        )
    return outputs

def evaluate_perf(
    device_preds: List[torch.Tensor],
    device_gold_labels: List[torch.Tensor],
    device_loss: float = None,
    device_samples: int = None,
    prefix: str = "Train",
    **kwargs
):
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0 # main process for logging, checkpointing, etc.

    agg_loss = None
    if device_loss is not None and device_samples is not None:
        total_loss_tensor = torch.tensor(device_loss, device=device)
        total_samples_tensor = torch.tensor(device_samples, device=device)
        if ddp:
            # aggregate losses across devices
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        agg_loss = total_loss_tensor.item() / total_samples_tensor.item()

    preds = torch.cat(device_preds, dim=0)
    gold_labels = torch.cat(device_gold_labels, dim=0)
    # Gather predictions from all processes
    if ddp:
        all_preds = [torch.zeros_like(preds) for _ in range(ddp_world_size)]
        all_gold_labels = [torch.zeros_like(gold_labels) for _ in range(ddp_world_size)]
        dist.all_gather(all_preds, preds)
        dist.all_gather(all_gold_labels, gold_labels)
        preds = torch.cat(all_preds, dim=0)
        gold_labels = torch.cat(all_gold_labels, dim=0)
    
    preds_np = preds.cpu().numpy()
    gold_labels_np = gold_labels.cpu().numpy()
    cur_perf = None
    if master_process:
        accuracy = accuracy_score(gold_labels_np, preds_np)
        prec_w, recall_w, f1_w, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='weighted')
        prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='micro')
        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='macro')

        cur_perf = PerfMetrics(
            epoch=kwargs.get("epoch", 9999),
            step=kwargs.get("step", 9999),
            accuracy=accuracy,
            precision={
                "weighted": prec_w,
                "micro": prec_micro,
                "macro": prec_macro
            },
            recall={
                "weighted": recall_w,
                "micro": recall_micro,
                "macro": recall_macro
            },
            f1={
                "weighted": f1_w,
                "micro": f1_micro,
                "macro": f1_macro
            },
            loss=agg_loss
        )

        if agg_loss is not None:
            logging.info(f"{prefix} loss: {agg_loss:.10f}")
        logging.info(f"{prefix} accuracy: {accuracy:.10f}")
        logging.info(f"{prefix} weighted precision: {prec_w:.10f}, recall: {recall_w:.10f}, f1: {f1_w:.10f}")
        logging.info(f"{prefix} micro precision: {prec_micro:.10f}, recall: {recall_micro:.10f}, f1: {f1_micro:.10f}")
        logging.info(f"{prefix} macro precision: {prec_macro:.10f}, recall: {recall_macro:.10f}, f1: {f1_macro:.10f}")

    return cur_perf

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
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dist.barrier()

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
                        dtype=torch.bfloat16, device=training_args.device  # pyre-fixme
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
        if model_args.tune_cls_head:
            for p in model.cls_head.parameters():
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
    
    model.to(torch.bfloat16)
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

    num_epochs: int = training_args.num_train_epochs
    last_epoch: int = 0
    batch_in_last_epoch: int = -1
    global_steps: int = 0
    gradient_accumulation_steps: int = int(training_args.gradient_accumulation_steps)
    # Create a reproducible generator
    generator = torch.Generator()
    if training_args.resume_from_checkpoint is not None:
        checkpoint_base_dir = os.path.dirname(training_args.resume_from_checkpoint)
        assert os.path.isdir(checkpoint_base_dir), f"Checkpoint base dir {checkpoint_base_dir} does not exist"
        assert os.path.isfile(training_args.resume_from_checkpoint), f"Checkpoint {training_args.resume_from_checkpoint} is not a file"
        ckpt = load_checkpoint(
            training_args.resume_from_checkpoint, 
            training_args,
            model, 
            load_optimizer=True, 
            load_scheduler=True
        )
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
        generator.set_state(ckpt["sampler_state"])
        num_warmup_steps = ckpt["num_warmup_steps"]
        num_training_steps = ckpt["num_training_steps"]
    else:
        generator.manual_seed(GLOBAL_SEED)
    
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
    os.makedirs(training_args.output_dir, exist_ok=True)
    train_log_fpath = os.path.join(training_args.output_dir, training_args.train_log)
    train_perf_log_fpath = os.path.join(training_args.output_dir, training_args.train_perf_log)
    eval_perf_log_fpath = os.path.join(training_args.output_dir, training_args.eval_perf_log)
    train_logs: List[TrainProgressLog] = []
    train_perf: List[PerfMetrics] = []
    eval_perf: List[PerfMetrics] = []

    for epoch in range(from_epoch, num_epochs):
        if ddp:
            # Ensure each process sees a different ordering at each epoch
            if epoch > last_epoch:
                generator.manual_seed(GLOBAL_SEED + epoch)
        model.train()
        train_loss_accum = torch.zeros(1, device=device)
        train_device_preds, train_device_gold_labels = [], []
        for batch_idx, batch in enumerate(train_dataloader):
            if epoch == from_epoch and batch_idx < from_batch:
                continue
            log_rank0(f'Epoch {epoch + 1}/{num_epochs}, batch {batch_idx + 1}/{len(train_dataloader)}')

            is_last_micro = ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx == len(train_dataloader) - 1)
            
            # DDP: skip gradient synchronisation on all but final micro-step
            ddp_context = model.no_sync() if (ddp and not is_last_micro) else nullcontext()
            train_labels = batch["eng_classes"].to(device)
            with ddp_context:
                outputs = forward_step(model, batch, device)
                cur_preds = torch.argmax(outputs.cls_logits, dim=-1)
                train_device_preds.append(cur_preds)
                train_device_gold_labels.append(train_labels)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                train_loss_accum += loss.detach()
                loss.backward()
            
            if is_last_micro:
                #   Update weights every accum_steps mini-batches
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
                        learning_rate=optimizer.param_groups[0]["lr"]
                    ))
                    with open(train_log_fpath, "w") as f:
                        json_train_logs = [log.to_dict() for log in train_logs]
                        json.dump(json_train_logs, f, indent=4)
                    # for param_group in optimizer.param_groups:
                    #     cur_lr = param_group["lr"]
                    #     logging.info(f'lr: {cur_lr:.10f}')
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
                    train_perf_log = evaluate_perf(
                        device_preds=train_device_preds,
                        device_gold_labels=train_device_gold_labels,
                        prefix="Train",
                        epoch=epoch + (batch_idx+1) / len(train_dataloader),
                        step=global_steps
                    )
                    if train_perf_log is not None:
                        # only on master process
                        train_perf.append(train_perf_log)
                        with open(train_perf_log_fpath, "w") as f:
                            json_train_perf = [perf.to_dict() for perf in train_perf]
                            json.dump(json_train_perf, f, indent=4)

                    if ddp:
                        dist.barrier() # wait for all processes to finish before evaluation
                    model.eval()
                    
                    eval_device_loss = 0.
                    eval_device_samples = 0
                    eval_device_preds, eval_device_gold_labels = [], []
                    for eval_batch_idx, eval_batch in enumerate(eval_dataloader):
                        log_rank0(f'After epoch {epoch + 1}, eval batch {eval_batch_idx+1}/{len(eval_dataloader)}')

                        eval_labels = eval_batch["eng_classes"].to(device)

                        with torch.no_grad():
                            outputs = forward_step(model, eval_batch, device, eval_mode=True)
                            eval_logits = outputs.cls_logits
                            cur_preds = torch.argmax(eval_logits, dim=-1)
                            eval_device_preds.append(cur_preds)
                            eval_device_gold_labels.append(eval_labels)
                            loss_fnc = torch.nn.CrossEntropyLoss()
                            eval_loss = loss_fnc(eval_logits, eval_labels)
                            eval_device_loss += eval_loss.item() * eval_labels.shape[0]
                            eval_device_samples += eval_labels.shape[0]
                    
                    eval_perf_log = evaluate_perf(
                        device_loss=eval_device_loss,
                        device_samples=eval_device_samples,
                        device_preds=eval_device_preds,
                        device_gold_labels=eval_device_gold_labels,
                        prefix="Eval",
                        epoch=epoch + (batch_idx+1) / len(train_dataloader),
                        step=global_steps
                    )
                    if eval_perf_log is not None:
                        # only on master process
                        eval_perf.append(eval_perf_log)
                        with open(eval_perf_log_fpath, "w") as f:
                            json_eval_perf = [perf.to_dict() for perf in eval_perf]
                            json.dump(json_eval_perf, f, indent=4)
                    model.train()
    
                do_save = False
                checkpoint_name = model_args.checkpoint_fname
                if training_args.save_strategy == 'epoch':
                    do_save = (batch_idx == len(train_dataloader) - 1)
                    if do_save:
                        checkpoint_name = f'{checkpoint_name}-epoch{epoch}.pt'
                elif training_args.save_strategy == 'steps':
                    do_save = (global_steps % eval_steps == 0)
                    if do_save:
                        checkpoint_name = f'{checkpoint_name}-epoch{epoch}-step{global_steps}.pt'
                if not do_save and epoch == num_epochs - 1 and batch_idx == len(train_dataloader) - 1:
                    # always save checkpoint at the very last training step
                    do_save = True
                    checkpoint_name = f'{checkpoint_name}-epoch{epoch}-final.pt'
                if master_process and do_save:
                    log_rank0(f'Saving checkpoint at epoch {epoch}, global step {global_steps}...')
                    checkpoint_path = os.path.join(training_args.output_model_filename, checkpoint_name)
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
                        sampler_gen_state=generator.get_state(),
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        per_device_train_batch_size=training_args.per_device_train_batch_size,
                        world_size=ddp_world_size,
                    )
        
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    # os.makedirs(SAFETENSORS_PATH, exist_ok=True)
    # mp.set_start_method('spawn')
    train()
