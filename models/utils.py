import os
import logging
import torch
import numpy as np
import random
import hashlib


def log_rank0(message: str):
    if int(os.environ.get("RANK", 0)) == 0:
        logging.info(message)

def count_parameters(model, print_layers = False):
    """
    Print the number of frozen and trainable parameters in the model,
    along with each layer's name, dtype, and trainability status.
    """
    frozen_params = 0
    trainable_params = 0
    trainable_layers = []

    if print_layers:
        # Print header for layer details
        print(f"{'Layer Name':<40} {'Dtype':<15} {'Trainable':<10} {'Param #':<15}")
        print("="*80)

    # Iterate over all named parameters in the model
    for name, param in model.named_parameters():
        param_count = param.numel()
        is_trainable = param.requires_grad
        dtype = str(param.dtype)

        if print_layers:
            # Print layer details
            print(f"{name:<40} {dtype:<15} {str(is_trainable):<10} {param_count:<15}")

        # Accumulate parameter counts
        if is_trainable:
            trainable_params += param_count
            trainable_layers.append(name)
        else:
            frozen_params += param_count

    backbone_params = {
        "siglip": 0,
        "dinov2": 0,
        "sva": 0,
        "projection": 0,
        "llm": 0
    }
    for name, param in model.named_parameters():
        if "vision_tower_aux_list.0" in name:
            # SigLIP
            backbone_params["siglip"] += param.numel()
        if "vision_tower_aux_list.1" in name:
            # DINOv2
            backbone_params["dinov2"] += param.numel()
        if "mm_projector" in name:
            # projection
            backbone_params["projection"] += param.numel()
        if "vision_sampler" in name or "vision_query" in name or "image_newline" in name:
            # sva
            backbone_params["sva"] += param.numel()
        if "model.layers" in name or "model.norm" in name or "model.embed_tokens" in name or "lm_head.weight" == name:
            # LLM
            backbone_params["llm"] += param.numel()
    print(f"{'Backbone Parameters':<40} {'':<15} {'':<10} {'':<15}")
    for key, value in backbone_params.items():
        print(f"{key:<40} {'':<15} {'':<10} {value:<15}")
    print("="*80)

    # Print summary
    print("="*80)
    print(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {frozen_params + trainable_params}")
    if print_layers:
        print("\nTrainable layers:")
        for layer in trainable_layers:
            print(f"- {layer}")


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


def gen_hex(gen: torch.Generator) -> str:
    return hashlib.sha1(gen.get_state().cpu().numpy().tobytes()).hexdigest()[:12]

class QualitativeSample:

    def __init__(
        self,
        video_path: str,
        cls_pred: int,
        gold_label: int,
        loss: float,
    ):
        self.video_path = video_path
        self.cls_pred = cls_pred
        self.gold_label = gold_label
        self.loss = loss

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "cls_pred": self.cls_pred,
            "gold_label": self.gold_label,
            "loss": self.loss,
        }

class TopKSelector:

    def __init__(
        self, 
        k: int,
        init_instances: list = None,
        eval_func: callable = None,
    ):
        # assert to check that eval_func must be provided and callable
        if eval_func is None:
            raise ValueError("eval_func must be provided and callable")
        if not callable(eval_func):
            raise ValueError("eval_func must be a callable function")
        if init_instances is not None:
            assert type(init_instances) == list or type(init_instances) == tuple, "init_instances must be a list or tuple"                
            for i in range(1, len(init_instances)):
                if eval_func(init_instances[i]) > eval_func(init_instances[i-1]):
                    raise ValueError("init_instances must be sorted in non-increasing order according to eval_func")
        
        self.k = k
        self.instances = init_instances if init_instances is not None else []
        self.eval_func = eval_func

    def add(self, instance) -> bool:
        if len(self.instances) > 0 and type(instance) != type(self.instances[0]):
            raise TypeError("instance must be of the same type as instances in the selector")
        is_inserted = False
        for i in range(len(self.instances)):
            if self.eval_func(instance) >= self.eval_func(self.instances[i]):
                # insert instance at position i
                self.instances.insert(i, instance)
                self.instances = self.instances[:self.k]
                is_inserted = True
                break
        if is_inserted is False and len(self.instances) < self.k:
            self.instances.append(instance)
            is_inserted = True
        return is_inserted

    def __len__(self):
        return len(self.instances)

    def get_instances(self):
        return self.instances