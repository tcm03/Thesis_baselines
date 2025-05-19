import os
import torch
import logging
from models.utils import log_rank0

def load_model(model, model_path):
    pretrained_state_dict = torch.load(model_path, map_location = 'cpu')
    model_state_dict = model.state_dict()
    log_rank0("Initializing layers...")
    for layer_name in model_state_dict.keys():
        if "vision_tower" in layer_name:
            # skip manual loading of vision encoders
            continue
        load_pretrained = False
        for pretrained_layer_name in pretrained_state_dict.keys():
            if layer_name in pretrained_layer_name:
                log_rank0(f"Load {layer_name}")
                model_state_dict[layer_name] = pretrained_state_dict[pretrained_layer_name]
                load_pretrained = True
                break
        if load_pretrained is False:
            log_rank0(f"Skip {layer_name}")
    model.load_state_dict(model_state_dict)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None
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
            log_rank0(f"Key {k} not found in model state dict.")
    raw_model.load_state_dict(current_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"] + 1
        log_rank0(f"Resuming training from epoch {epoch}")
    else:
        epoch = 0
        log_rank0("No epoch information found in checkpoint. Starting from epoch 0.")
    if "global_steps" in checkpoint:
        global_steps = checkpoint["global_steps"]
        log_rank0(f"Resuming training from global step {global_steps + 1}")
    else:
        global_steps = 0
        log_rank0("No global step information found in checkpoint. Starting from global step 0.")
    return epoch, global_steps

def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler = None,
    epoch: int = None,
    global_steps: int = None
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
    log_rank0(f"Checkpoint saved to {checkpoint_path}")
