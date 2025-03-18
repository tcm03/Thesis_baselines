import sys
sys.path.append('.')

import torch
import os
import argparse
from typing import List, Dict
from mm_datautils import process_video_frames
from preprocessor import CambrianConfig, CambrianMeta
from safetensors.torch import save_file
from collections import defaultdict
import logging
from multiprocessing import cpu_count
from engagement_dataset import EngagementDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import BaseImageProcessor
from constants import *

# import annotation.utils (which imports decord) after torch to avoid bug
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from annotation.utils import get_optimal_workers
import torch.multiprocessing as mp
from resource_logging import measure_resource_usage, MeasureResourceUsage

# Configure logging with line numbers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

def extract_fileid(file_path: str) -> str:
    return file_path.split('.')[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        required=True,
        help="List of folder paths to video data"
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

    # Set up device (and log it)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    cambrianConfig = CambrianConfig.from_json_file(args.config_file)
    model = CambrianMeta(cambrianConfig)

    for vision_tower_aux in model.vision_tower_aux_list:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()

    model = model.to(device)
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
    

    folder_paths: List[str] = args.data
    entube_dataset = EngagementDataset(folder_paths, image_processors)
    dataloader = DataLoader(
        entube_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
    )

    # model.eval()

    for batch_idx, (videos, image_sizes, file_names) in enumerate(dataloader):
        logging.info(f'Processing batch {batch_idx + 1}/{len(dataloader)}')
        assert isinstance(videos, list), "List of videos features for each processor (vision encoder)"
        assert isinstance(videos[0], list) or isinstance(videos[0], torch.Tensor), "List of videos in the batch"
        # tensor(num_reduced_frames, len=576, hidden_dim=1152/1536) image_aux_features_list[num_processors]

        pred_logits = model(videos, image_sizes)

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
