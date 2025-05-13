import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR # for verifying the correctness of checkpoint resumption only, no real use

import os
import json
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

# baseline libs
from mm_datautils import process_video_frames
from utils import *
from preprocessor import CambrianConfig, CambrianMeta

from collections import defaultdict
import logging
from multiprocessing import cpu_count
from constants import *



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

def load_model(model, model_path, master_process=False):
    pretrained_state_dict = torch.load(model_path, map_location = 'cpu')
    model_state_dict = model.state_dict()
    if master_process:
        logging.info("Initializing layers...")
    for layer_name in model_state_dict.keys():
        if "vision_tower" in layer_name:
            # skip manual loading of vision encoders
            continue
        load_pretrained = False
        for pretrained_layer_name in pretrained_state_dict.keys():
            if layer_name in pretrained_layer_name:
                if master_process:
                    logging.info(f"Load {layer_name}")
                model_state_dict[layer_name] = pretrained_state_dict[pretrained_layer_name]
                load_pretrained = True
                break
        if load_pretrained is False and master_process is True:
            logging.info(f"Skip {layer_name}")
    model.load_state_dict(model_state_dict)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    master_process: bool = False
):
    """
    Loads a checkpoint containing only the trainable parts of the model,
    along with the optimizer and scheduler states.

    Args:
        checkpoint_path (str): File path to the checkpoint.
        model (torch.nn.Module): The model (can be wrapped in DDP).
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (optional): Learning rate scheduler; its state is loaded if provided.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_model = model.module if hasattr(model, "module") else model
    current_state_dict = raw_model.state_dict()
    for k, v in checkpoint["model_trainable_state_dict"].items():
        if k in current_state_dict:
            current_state_dict[k].copy_(v)
        else:
            if master_process is True:
                logging.warning(f"Key {k} not found in model state dict.")
    raw_model.load_state_dict(current_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"] + 1
        if master_process is True:
            logging.info(f"Resuming training from epoch {epoch}")
    else:
        epoch = 0
        if master_process is True:
            logging.info("No epoch information found in checkpoint. Starting from epoch 0.")
    if "global_steps" in checkpoint:
        global_steps = checkpoint["global_steps"]
        if master_process is True:
            logging.info(f"Resuming training from global step {global_steps + 1}")
    else:
        global_steps = 0
        if master_process is True:
            logging.info("No global step information found in checkpoint. Starting from global step 0.")
    return epoch, global_steps

def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler = None,
    epoch: int = None,
    global_steps: int = None,
    master_process: bool = False
):
    """
    Saves a checkpoint containing only the trainable parts of the model,
    along with the optimizer and scheduler states.
    
    This is useful when most parameters are frozen (e.g., vision encoders, 
    spatial aggregators, vision queries) and only a subset (e.g., last FC layers
    and layer norm layers) is being fine-tuned. 
    
    Args:
        checkpoint_path (str): File path to save the checkpoint.
        model (torch.nn.Module): The model (can be wrapped in DDP).
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (optional): Learning rate scheduler; its state is saved if provided.
        epoch (int, optional): Current epoch number.
        global_steps (int, optional): Total number of training steps completed.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    trainable_keys = [n for n, t in raw_model.named_parameters() if t.requires_grad]
    trainable_state_dict = {k: v for k, v in raw_model.state_dict().items() if k in trainable_keys}
    checkpoint = {
        "model_trainable_state_dict": trainable_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if global_steps is not None:
        checkpoint["global_steps"] = global_steps
    torch.save(checkpoint, checkpoint_path)
    if master_process is True:
        logging.info(f"Checkpoint saved to {checkpoint_path}")


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
    print(f"Using conversation format: {conversation_lib.default_conversation.version}")
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
    
    model.to(torch.bfloat16)
    # pyre-fixme
    def convert_bn_to_float(model):
        if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
            return model.float()
        for child_name, child in model.named_children():
            model.add_module(child_name, convert_bn_to_float(child))
        return model

    model = convert_bn_to_float(model)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    eval_dataset = data_module["eval_dataset"]
    data_collator = data_module["data_collator"]

    assert training_args.group_by_modality_length is True, "Group by modality length must be True"
    # Create a reproducible generator
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)
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
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = get_optimizer(model, training_args)

    num_epochs: int = training_args.num_train_epochs
    start_epoch: int = 0
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)  # warm up % of training steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    global_steps: int = 0
    # if training_args.resume:
    #     assert custom_args.input_checkpoint_path is not None, "Checkpoint path is required for resuming training"
    #     ( 
    #         start_epoch, 
    #         global_steps
    #     ) = load_checkpoint(
    #         training_args.input_checkpoint_path,
    #         raw_model,
    #         optimizer,
    #         scheduler=scheduler,
    #         master_process=master_process
    #     )

    logging_steps: int = int(training_args.logging_steps)
    eval_steps: int = int(training_args.eval_steps)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if ddp:
            # Ensure each process sees a different ordering at each epoch
            # train_sampler.set_epoch(epoch)
            generator.manual_seed(GLOBAL_SEED + epoch)
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            if master_process:
                logging.info(f'Epoch {epoch + 1}/{start_epoch + num_epochs}, batch {batch_idx + 1}/{len(train_dataloader)}')

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            image_aux_attention_masks_list = [image_aux_attn_mask.to(device) for image_aux_attn_mask in batch["image_aux_attention_masks_list"]]
            image_sizes = batch["image_sizes"]
            images = None
            if "images" in batch:
                images = [image.to(device) for image in batch["images"]]

            optimizer.zero_grad()

            # pred_logits = model(input_ids, attention_mask, position_ids, labels, images, image_aux_attention_masks_list, image_sizes)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                images=images,
                image_aux_attention_masks_list=image_aux_attention_masks_list,
                image_sizes=image_sizes,
            )
            loss = outputs[0]
            logits = outputs[1]
            if master_process:
                logging.info(f'loss: {loss.item()}, logits: {logits}')
            continue # testing
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize() # wait for the GPU to finish work
            global_steps += 1

            if global_steps % logging_steps == 0 and master_process:
                total_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                logging.info(f'Epoch {epoch + 1}/{start_epoch + num_epochs}, global step: {global_steps}, loss={loss.item()}, clipped gradient norm: {total_norm:.4f}')
                for param_group in optimizer.param_groups:
                    cur_lr = param_group["lr"]
                    logging.info(f'lr: {cur_lr:.10f}')

            do_eval = False
            if training_args.eval_strategy == 'epoch':
                do_eval = (batch_idx == len(train_dataloader) - 1)
            elif training_args.eval_strategy == 'steps':
                do_eval = (global_steps % eval_steps == 0)
            # If `"epoch"` or `"steps"` is chosen, saving will also be performed at the very end of training, always.
            if epoch == num_epochs - 1 and batch_idx == len(train_dataloader) - 1:
                do_eval = True
            if do_eval:
                if ddp:
                    dist.barrier() # wait for all processes to finish before evaluation
                model.eval()
                
                device_loss = 0.
                device_samples = 0
                device_preds, device_gold_labels = [], []
                for eval_batch_idx, (eval_filenames, eval_videos, eval_image_sizes, eval_labels) in enumerate(eval_dataloader):
                    if master_process:
                        logging.info(f'After epoch {epoch + 1}, eval batch {eval_batch_idx+1}/{len(eval_dataloader)}')
                    for i, videos_aux in enumerate(eval_videos):
                        eval_videos[i] = [video.to(device) for video in videos_aux]
                    eval_labels = eval_labels.to(device)

                    with torch.no_grad():
                        eval_logits = model(eval_videos, eval_image_sizes)
                        cur_preds = torch.argmax(eval_logits, dim=-1)
                        device_preds.append(cur_preds)
                        device_gold_labels.append(eval_labels)
                        eval_loss = loss_fnc(eval_logits, eval_labels)
                        device_loss += eval_loss.item() * eval_labels.shape[0]
                        device_samples += eval_labels.shape[0]
                
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
                
                # if master_process:
                #     logging.info(f"len(preds): {len(preds)}, len(gold_labels): {len(gold_labels)}")
                preds_np = preds.cpu().numpy()
                gold_labels_np = gold_labels.cpu().numpy()
                if master_process:
                    accuracy = accuracy_score(gold_labels_np, preds_np)
                    prec_w, recall_w, f1_w, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='weighted')
                    prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='micro')
                    prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='macro')

                    logging.info(f"Evaluation loss: {agg_loss:.10f}")
                    logging.info(f"Evaluation accuracy: {accuracy:.10f}")
                    logging.info(f"Evaluation weighted precision: {prec_w:.10f}, recall: {recall_w:.10f}, f1: {f1_w:.10f}")
                    logging.info(f"Evaluation micro precision: {prec_micro:.10f}, recall: {recall_micro:.10f}, f1: {f1_micro:.10f}")
                    logging.info(f"Evaluation macro precision: {prec_macro:.10f}, recall: {recall_macro:.10f}, f1: {f1_macro:.10f}")
                model.train()
    
            do_save = False
            checkpoint_name = None
            if training_args.save_strategy == 'epoch':
                do_save = (batch_idx == len(train_dataloader) - 1)
                if do_save:
                    checkpoint_name = f'checkpoint-epoch{epoch}.pt'
            elif training_args.save_strategy == 'steps':
                do_save = (global_steps % eval_steps == 0)
                if do_save:
                    checkpoint_name = f'checkpoint-epoch{epoch}-steps{global_steps}.pt'
            # if do_save:
            #     if master_process:
            #         logging.info(f'Saving checkpoint at epoch {epoch}, global step {global_steps}...')
            #     checkpoint_path = os.path.join(training_args.output_checkpoint_path, checkpoint_name)
            #     save_checkpoint(
            #         checkpoint_path,
            #         raw_model,
            #         optimizer,
            #         # scheduler=scheduler,
            #         # epoch=epoch,
            #         # global_steps=global_steps,
            #         master_process=master_process
            #     )

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    # os.makedirs(SAFETENSORS_PATH, exist_ok=True)
    # mp.set_start_method('spawn')
    train()
