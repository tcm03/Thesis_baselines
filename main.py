import sys
sys.path.append('.')

import torch
import os
import argparse
from typing import List, Dict
from mm_datautils import process_video_frames
from utils import *
from preprocessor import CambrianConfig, CambrianMeta
from safetensors.torch import save_file
from collections import defaultdict
import logging
from multiprocessing import cpu_count
from engagement_dataset import EngagementDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import BaseImageProcessor
from constants import *

from transformers import get_cosine_schedule_with_warmup

# import annotation.utils (which imports decord) after torch to avoid bug
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
import torch.multiprocessing as mp
from resource_logging import measure_resource_usage, MeasureResourceUsage

# Configure logging with line numbers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

def load_model(model, model_path):
    pretrained_state_dict = torch.load(model_path, map_location = 'cpu')
    model_state_dict = model.state_dict()
    for layer_name in model_state_dict.keys():
        if "vision_tower" in layer_name:
            # skip manual loading of vision encoders
            continue
        for pretrained_layer_name in pretrained_state_dict.keys():
            if layer_name in pretrained_layer_name:
                model_state_dict[layer_name] = pretrained_state_dict[pretrained_layer_name]
                break
            
    model.load_state_dict(model_state_dict)

def train(args):
    # Set up device (and log it)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    cambrianConfig = CambrianConfig.from_json_file(args.config_file)
    model = CambrianMeta(cambrianConfig)
    load_model(model, args.model_path)

    for vision_tower_aux in model.vision_tower_aux_list:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()

    model = model.to(device)
    for n, t in model.named_parameters():
        if "last_fc" in n:
            t.requires_grad = True
        else:
            t.requires_grad = False

    logging.info(f"Model state dict\n{model.state_dict()}")
    logging.info("Trainable layers")
    for n, t in model.named_parameters():
        if t.requires_grad:
            logging.info(f'Layer name: {n}')
    # For inference on multiple GPUs, wrap with DataParallel if more than one GPU is available.
    # if torch.cuda.device_count() > 1:
    #     logging.info(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
    #     processor = torch.nn.DataParallel(processor)
    # # When using DataParallel, submodules are under processor.module.
    # model_module = processor.module if isinstance(processor, torch.nn.DataParallel) else processor

    image_processors = []
    # for vision_tower_aux in model_module.vision_tower_aux_list:
    for vision_tower_aux in model.vision_tower_aux_list:
        image_processors.append(vision_tower_aux.image_processor)
    

    engagement_dataset = EngagementDataset(args.data_path, args.json_path, image_processors, device)
    dataloader = DataLoader(
        engagement_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
    )
    # AdamW optimizer for training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-6,         # Learning rate
        weight_decay=0.0 # Weight decay
    )
    loss_fnc = torch.nn.CrossEntropyLoss()

    # model.eval()
    model.train()
    num_epochs: int = 2
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.03 * num_training_steps)  # 3% of training steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )   

    logging_steps: int = 1
    global_steps: int = 0
    for epoch in range(num_epochs):

        for batch_idx, (filenames, videos, image_sizes, labels) in enumerate(dataloader):
            logging.info(f'Processing batch {batch_idx + 1}/{len(dataloader)}')
            assert isinstance(videos, list), "List of videos features for each processor (vision encoder)"
            assert isinstance(videos[0], list) or isinstance(videos[0], torch.Tensor), "List of videos in the batch"
            assert isinstance(image_sizes, list) or isinstance(image_sizes, tuple), "List/Tuple of frame sizes of videos in the batch"
            # tensor(num_reduced_frames, len=576, hidden_dim=1152/1536) image_aux_features_list[num_processors]

            optimizer.zero_grad()
            pred_logits = model(videos, image_sizes)
            loss = loss_fnc(pred_logits, labels)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logging.info(f"Global step: {global_steps}, pred_logits: {pred_logits}, labels: {labels}, gradient norm: {total_norm:.4f}")

            optimizer.step()
            scheduler.step()
            global_steps += 1

            if global_steps % logging_steps == 0:
                logging.info(f'Epoch {epoch + 1}/{num_epochs}, global step {global_steps}: loss={loss.item()}')


        # tensor_siglip = image_aux_features_list[0].to('cpu')
        # tensor_dino = image_aux_features_list[1].to('cpu')
        # # file_name = file_names[0] # the batch has only one file
        # for file_name in file_names:
        #     logging.info(f'file_name={file_name}')
        #     file_id = extract_fileid(file_name)
        #     save_tensor = {
        #         file_id + '-siglip': tensor_siglip,
        #         file_id + '-dino': tensor_dino
        #     }
        #     safetensors_file_path = os.path.join(SAFETENSORS_PATH, file_id + '.safetensors')
        #     save_file(save_tensor, safetensors_file_path)
            
        #     # Get the file size
        #     try:
        #         file_size = os.path.getsize(safetensors_file_path)
        #         logging.info(f"Safetensors file '{safetensors_file_path}' size: {file_size / (1024 * 1024):.2f} MB")
        #     except FileNotFoundError:
        #         logging.warning(f"Safetensors file '{safetensors_file_path}' not found after saving.")
        #         continue

        #     # Delete the file after evaluating its size
        #     try:
        #         os.remove(safetensors_file_path)
        #         logging.info(f"Safetensors file '{safetensors_file_path}' deleted successfully.")
        #     except OSError as e:
        #         logging.error(f"Error deleting file '{safetensors_file_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to video dataset"
    )
    parser.add_argument(
        '--train_path',
        type=str,
        required=True,
        help="Path to metadata file of the training video dataset"
    )
    parser.add_argument(
        '--eval_path',
        type=str,
        required=True,
        help="Path to metadata file of the evaluation video dataset"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        '--output_file',
        type = str,
        default = 'entube_tensors.safetensors',
        help = 'Safetensor file to store embeddings of EnTube dataset by vision encoders'
    )
    parser.add_argument(
        '--config_file',
        type = str,
        default = 'config.json',
        help = 'Path to configuration file of encoders parameters'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (global batch size, will be split across GPUs)'
    )
    args = parser.parse_args()
    os.makedirs(SAFETENSORS_PATH, exist_ok=True)
    # mp.set_start_method('spawn')

    train(args)
