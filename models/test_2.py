import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import transformers

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

is_distributed = int(os.environ.get("RANK", -1)) != -1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

GLOBAL_SEED = 1337
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)

def forward_step(
    model, 
    batch, 
    device,
    model_args,  
    tokenizer,
    eval_mode=False, 
    cls_only=False,
    cls_loss_weight=None,
    gen_config_dict: Dict[str, Any]=None
):
    if gen_config_dict is not None and cls_only:
        raise ValueError("gen_config_dict for text generation is not supported for cls_only")
    # if gen_config_dict is not None and not eval_mode:
    #     raise ValueError("gen_config_dict for text generation is only supported for eval_mode")
    
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
        if cls_only:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                # labels=labels,
                eng_classes=eng_classes,
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
                cls_loss_weight=cls_loss_weight,
                images=images,
                image_aux_attention_masks_list=image_aux_attention_masks_list,
                image_sizes=image_sizes,
            )
    if not cls_only and gen_config_dict is not None:
        # @tcm: add gen_config_dict to model.generate()
        conv = conversation_lib.conv_templates[model_args.version].copy()
        stop_str = conv.sep if conv.sep_style != conversation_lib.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        raw = model
        # attention_mask = ()
        with torch.inference_mode():
            was_training = raw.training
            if was_training:
                raw.eval() # disable dropout and checkpointing (use_cache can be True)
            output_ids = raw.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=image_sizes,
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
        # log_rank0(f"In forward_step(): video_path: {batch['video_path']}, pred: {pred}")
        outputs["preds"] = [pred]
    return outputs

def inference():
    
    if is_distributed:
        # Distributed inference
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0 # main process for logging, checkpointing, etc.
    else:
        # non-distributed
        rank = 0
        local_rank = 0
        world_size = 1
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
    if is_distributed:
        dist.barrier()

    if training_args.bf16 and training_args.fp16:
        raise ValueError("Cannot use both bf16 and fp16")

    # pyre-fixme[16]: `DataClass` has no attribute `output_model_local_path`.
    training_args.output_dir = model_args.output_model_filename
    # pyre-fixme[16]: `DataClass` has no attribute `local_dir`.
    model_args.local_dir = model_args.output_model_filename
    # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute `from_pretrained`.
    
    if model_args.cls_only:
        model = CambrianLlamaForSequenceClassification.from_pretrained(
            model_args.input_model_filename,
        )
    else:
        model = CambrianLlamaForCausalLM.from_pretrained(
            # pyre-fixme[16]: `DataClass` has no attribute `input_model_local_path`.
            model_args.input_model_filename,
        )
    model.config.use_cache = False
    # pyre-fixme[16]: `DataClass` has no attribute `freeze_backbone`.
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    # pyre-fixme[16]: `DataClass` has no attribute `gradient_checkpointing`.
    # [2025-07-16] @tcm: I'm trying to hide this during inference, could change
    # if training_args.gradient_checkpointing:
    #     # @tcm: might look here: https://junbuml.ee/grad-flow-lora-grad-ckpt
    #     model.config.use_cache = False # Disable KV-cache (mandatory with ckpt)
    #     model.gradient_checkpointing_enable()
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         # pyre-fixme[3]: Return type must be annotated.
    #         # pyre-fixme[2]: Parameter must be annotated.
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)

    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
        # [2025-07-16] @tcm: I freeze the entire model for inference
        model.requires_grad_(False)
        # if model_args.tune_mm_mlp_adapter:
        #     model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            # tune_modules = [
            #     "mm_projector",
            #     "pos_emb",
            #     "vision_sampler",
            #     "vision_sampler_layers",
            #     "vision_query",
            #     "image_newline",
            # ]
            # for name, param in model.named_parameters():
            #     if any(listed_name in name for listed_name in tune_modules):
            #         param.requires_grad = True
        # [2025-07-16] @tcm: I freeze the entire model for inference
        # if model_args.tune_lm_head and not model_args.cls_only:
        #     for p in model.lm_head.parameters():
        #         p.requires_grad = True
        # if model_args.tune_cls_head:
        #     if not model_args.cls_only:
        #         for p in model.cls_head.parameters():
        #             p.requires_grad = True
        #     else:
        #         for p in model.score.parameters():
        #             p.requires_grad = True
        # if model_args.tune_embed_tokens:
        #     for p in model.get_input_embeddings().parameters():
        #         p.requires_grad = True
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter  # pyre-fixme
        # if training_args.freeze_mm_mlp_adapter:
        #     for p in model.get_model().mm_projector.parameters():
        #         p.requires_grad = False
        # if training_args.unfreeze_mm_vision_tower:
        #     if vision_tower_aux_list is not None:
        #         for vision_tower_aux in vision_tower_aux_list:
        #             for p in vision_tower_aux.parameters():
        #                 p.requires_grad = True

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
            generator, # do not use during inference
            load_optimizer = False,
            load_scheduler = False
        )
        log_rank0("Loaded checkpoint")
    # For testing, we're not gonna care about checkpoint seed and use the default global seed instead
    generator.manual_seed(GLOBAL_SEED)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    eval_dataset = data_module["eval_dataset"]
    data_collator = data_module["data_collator"]

    assert training_args.group_by_modality_length is True, "Group by modality length must be True"
    # Instantiate LengthGroupedSampler
    eval_sampler = LengthGroupedSampler(
        batch_size=training_args.per_device_eval_batch_size,
        world_size=world_size,
        lengths=eval_dataset.modality_lengths,
        generator=generator,
        group_by_modality=training_args.group_by_modality_length,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True, # per-device eval batch size = 1 so we won't miss too many samples
    )
    
    if master_process:
        if os.path.exists(training_args.output_dir):
            raise ValueError(f"Output directory {training_args.output_dir} already exists")
        os.makedirs(training_args.output_dir, exist_ok=True)
    eval_perf_log_fpath = os.path.join(training_args.output_dir, training_args.eval_perf_log)
    eval_log_fpath = os.path.join(training_args.output_dir, training_args.eval_log)
    eval_perf: List[PerfMetrics] = []
    if training_args.generation_eval:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")

    log_rank0("Starting inference")
    if is_distributed:
        dist.barrier() # align all processes before evaluation
    model.eval()
    
    eval_device_loss = 0.
    eval_device_samples = 0
    eval_device_preds, eval_device_gold_labels = [], []
    eval_device_text_preds, eval_device_text_references = [], []
    eval_video_paths = []
    for eval_batch_idx, eval_batch in enumerate(eval_dataloader):
        log_rank0(f'eval batch {eval_batch_idx+1}/{len(eval_dataloader)}')

        eval_labels = eval_batch["eng_classes"].to(device)

        with torch.no_grad():
            outputs = forward_step(
                model, 
                eval_batch, 
                device, 
                model_args, 
                tokenizer, 
                eval_mode=True, 
                cls_only=model_args.cls_only,
                cls_loss_weight=training_args.cls_loss_weight,
                gen_config_dict={
                    "do_sample": False,
                    "max_new_tokens": 256,
                    "num_beams": 1,
                    "use_cache": True,
                } if training_args.generation_eval else None
            )
            eval_logits = outputs.cls_logits
            cur_preds = torch.argmax(eval_logits, dim=-1)
            eval_device_preds.append(cur_preds)
            eval_device_gold_labels.append(eval_labels)
            loss_fnc = torch.nn.CrossEntropyLoss()
            eval_loss = loss_fnc(eval_logits, eval_labels)
            eval_device_loss += eval_loss.item() * eval_labels.shape[0]
            eval_device_samples += eval_labels.shape[0]
            if training_args.generation_eval:
                eval_device_text_preds.extend(outputs["preds"])
                eval_device_text_references.extend(eval_batch["responses"])
            eval_video_paths.extend(eval_batch["video_paths"])

    if master_process and training_args.generation_eval:
        # logging.info(f"Eval video paths: {eval_video_paths}")
        # @tcm: At the moment, print out predicted label and generated text for each video in the eval set.
        assert len(eval_video_paths) == len(eval_device_text_preds) and len(eval_video_paths) == len(eval_device_preds), "need equal"
        eval_logs: List[EvalProgressLog] = []
        for video_path, cls_pred, gen_pred in zip(eval_video_paths, eval_device_preds, eval_device_text_preds):
            eval_logs.append(EvalProgressLog(
                video_path=video_path,
                cls_pred=cls_pred.item(),
                gen_pred=gen_pred
            ))
        cur_eval_log_fname = os.path.basename(eval_log_fpath).split(".")[0] + f"-test.json"
        cur_eval_log_fdir = os.path.dirname(eval_log_fpath)
        cur_eval_log_fpath = os.path.join(cur_eval_log_fdir, cur_eval_log_fname)
        with open(cur_eval_log_fpath, "w") as f:
            json_eval_logs = [log.to_dict() for log in eval_logs]
            json.dump(json_eval_logs, f, indent=4)

    eval_gathered_preds = [None for _ in range(world_size)] if master_process else None
    eval_gathered_references = [None for _ in range(world_size)] if master_process else None
    if training_args.generation_eval and is_distributed:
        dist.gather_object(eval_device_text_preds, eval_gathered_preds, dst=0)
        dist.gather_object(eval_device_text_references, eval_gathered_references, dst=0)
    if master_process and training_args.generation_eval:
        # flatten
        eval_gathered_preds = [pred for rank_preds in eval_gathered_preds for pred in rank_preds]
        eval_gathered_references = [ref for rank_refs in eval_gathered_references for ref in rank_refs]
    text_evaluators = {}
    if training_args.generation_eval:
        text_evaluators = {"bleu": bleu, "rouge": rouge, "meteor": meteor, "bertscore": bertscore}
    eval_perf_log = evaluate_perf(
        device_loss=eval_device_loss,
        device_samples=eval_device_samples,
        device_preds=eval_device_preds,
        device_gold_labels=eval_device_gold_labels,
        predictions=eval_gathered_preds if training_args.generation_eval else None,
        references=eval_gathered_references if training_args.generation_eval else None,
        prefix="Test",
        **text_evaluators
    )
    if eval_perf_log is not None:
        # only on master process
        eval_perf.append(eval_perf_log)
        with open(eval_perf_log_fpath, "w") as f:
            json_eval_perf = [perf.to_dict() for perf in eval_perf]
            json.dump(json_eval_perf, f, indent=4)
        
    if is_distributed:
        destroy_process_group()

if __name__ == "__main__":
    inference()
