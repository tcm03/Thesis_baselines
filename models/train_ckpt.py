# ===== models/train_ckpt.py =====
import os, torch, random, numpy as np
from typing import Dict, Any
from models.utils import log_rank0
from opti import get_optimizer
from transformers import get_cosine_schedule_with_warmup, TrainingArguments


def _rng_pack() -> Dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy":      np.random.get_state(),
        "python":     random.getstate(),
    }

def _rng_unpack(pkg: Dict[str, Any]):
    torch.set_rng_state(pkg["torch_cpu"])
    if pkg["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(pkg["torch_cuda"])
    np.random.set_state(pkg["numpy"])
    random.setstate(pkg["python"])

# ---------- SAVE ----------
def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    num_warmup_steps: int,
                    num_training_steps: int,
                    last_epoch: int,
                    batch_in_last_epoch: int,
                    global_steps: int,
                    sampler_gen_state,
                    gradient_accumulation_steps: int,
                    per_device_train_batch_size: int,
                    world_size: int,
                    ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = model.module if hasattr(model, "module") else model

    trainable = {k: v for k, v in raw.state_dict().items()
                 if v.requires_grad}

    torch.save({
        # model, optimizer, and scheduler
        "model_trainable_state_dict": trainable,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "num_warmup_steps": num_warmup_steps,
        "num_training_steps": num_training_steps,
        # training progress
        "last_epoch": last_epoch,
        "batch_in_last_epoch": batch_in_last_epoch,
        "global_steps": global_steps,
        # rng states
        "sampler_state": sampler_gen_state,
        "rng_state": _rng_pack(),
        # training config
        "world_size": world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "rank": int(os.environ.get("RANK", 0)),
    }, path)
    log_rank0(f"Checkpoint saved to {path}")

# ---------- LOAD ----------
def load_checkpoint(path: str,
                    training_args: TrainingArguments,
                    model: torch.nn.Module,
                    load_optimizer: bool = True,
                    load_scheduler: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    raw  = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_trainable_state_dict"], strict=False)
    return_dict = {}

    if load_optimizer:
        optimizer = get_optimizer(model, training_args)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return_dict["optimizer"] = optimizer

    if load_scheduler:
        if "optimizer" not in return_dict:
            raise ValueError("Optimizer must be loaded before scheduler")
        if "scheduler_state_dict" not in ckpt or "num_warmup_steps" not in ckpt or "num_training_steps" not in ckpt:
            raise ValueError("Scheduler state dict, num_warmup_steps, and num_training_steps must be provided if load_scheduler is True")

        last_scheduler_epoch = ckpt["scheduler_state_dict"]["last_epoch"]
        scheduler = get_cosine_schedule_with_warmup(
            return_dict["optimizer"],
            num_warmup_steps=ckpt["num_warmup_steps"],
            num_training_steps=ckpt["num_training_steps"],
            last_epoch=last_scheduler_epoch-1, # initialization automatically invokes step() so we subtract to avoid stepping twice
        )
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return_dict["scheduler"] = scheduler
        return_dict["num_warmup_steps"] = ckpt["num_warmup_steps"]
        return_dict["num_training_steps"] = ckpt["num_training_steps"]
    
    _rng_unpack(ckpt["rng_state"])

    return_dict.update({
        "last_epoch": ckpt["last_epoch"],
        "batch_in_last_epoch": ckpt["batch_in_last_epoch"],
        "global_steps": ckpt["global_steps"],
        "sampler_state": ckpt["sampler_state"],
        "world_size": ckpt["world_size"],
        "gradient_accumulation_steps": ckpt["gradient_accumulation_steps"],
        "per_device_train_batch_size": ckpt["per_device_train_batch_size"],
        "rank": int(os.environ.get("RANK", 0)),
    })

    return return_dict