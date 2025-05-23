import os
import logging
import torch
import numpy as np
import random

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