import sys
sys.path.append('.')

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

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

def train(args):
    ddp = int(os.environ.get("RANK", -1)) != -1
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

    cambrianConfig = CambrianConfig.from_json_file(args.config_file)
    model = CambrianMeta(cambrianConfig)
    load_model(model, args.model_path, master_process=master_process)

    for vision_tower_aux in model.vision_tower_aux_list:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()

    model.to(device)
    for n, t in model.named_parameters():
        if "fc_" in n or "ln_" in n:
            t.requires_grad = True
        else:
            t.requires_grad = False
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

#    logging.info(f"Model state dict")
#    for k, v in model.state_dict().items():
#        logging.info(f"{k}: {v.size()}")
    if master_process:
        logging.info("Trainable layers")
        for n, t in raw_model.named_parameters():
            if t.requires_grad:
                logging.info(f'Layer: {n}')
    # For inference on multiple GPUs, wrap with DataParallel if more than one GPU is available.
    # if torch.cuda.device_count() > 1:
    #     logging.info(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
    #     processor = torch.nn.DataParallel(processor)
    # # When using DataParallel, submodules are under processor.module.
    # model_module = processor.module if isinstance(processor, torch.nn.DataParallel) else processor

    image_processors = [vision_tower_aux.image_processor for vision_tower_aux in raw_model.vision_tower_aux_list]
    
    train_dataset = EngagementDataset(
        data_paths=args.data_paths, 
        json_paths=args.train_paths, 
        image_processors=image_processors
    )
    eval_dataset = EngagementDataset(
        data_paths=args.data_paths, 
        json_paths=args.eval_paths, 
        image_processors=image_processors
    )

    # Set up DistributedSampler (if in DDP mode) for automatic data splitting
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp else None
    # For evaluation we set shuffle=False to preserve order. Note that DistributedSampler pads samples,
    # so we will trim the predictions later.
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=True) if ddp else None

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        sampler=eval_sampler,
        drop_last=True,
        collate_fn=collate_fn,
    )
    # AdamW optimizer for training
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=5e-7,         # Learning rate
        weight_decay=0.0 # Weight decay
    )
    loss_fnc = torch.nn.CrossEntropyLoss()

    num_epochs: int = 1
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(0.03 * num_training_steps)  # 3% of training steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )   

    logging_steps: int = 10
    eval_steps: int = 380
    global_steps: int = 0
    for epoch in range(num_epochs):
        if ddp:
            # Ensure each process sees a different ordering at each epoch
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (filenames, videos, image_sizes, labels) in enumerate(train_dataloader):
            if master_process:
                logging.info(f'Epoch {epoch + 1}/{num_epochs}, batch {batch_idx + 1}/{len(train_dataloader)}')
            assert isinstance(videos, list), "List of videos features for each processor (vision encoder)"
            assert isinstance(videos[0], list) or isinstance(videos[0], torch.Tensor), "List of videos in the batch"
            assert isinstance(image_sizes, list) or isinstance(image_sizes, tuple), "List/Tuple of frame sizes of videos in the batch"
            # tensor(num_reduced_frames, len=576, hidden_dim=1152/1536) image_aux_features_list[num_processors]

            # videos: List[List[video_tensor1, ...], List[video_tensor1, ...]]
            for i, videos_aux in enumerate(videos):
                videos[i] = [video.to(device) for video in videos_aux]
            labels = labels.to(device)
            optimizer.zero_grad()

            pred_logits = model(videos, image_sizes)
            loss = loss_fnc(pred_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize() # wait for the GPU to finish work
            global_steps += 1

            if global_steps % logging_steps == 0 and master_process:
                total_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                logging.info(f'Epoch {epoch + 1}/{num_epochs}, global step: {global_steps}, loss={loss.item()}, clipped gradient norm: {total_norm:.4f}')
                for param_group in optimizer.param_groups:
                    cur_lr = param_group["lr"]
                    logging.info(f'lr: {cur_lr:.10f}')

            do_eval = (global_steps % eval_steps == 0) or (epoch == num_epochs - 1 and batch_idx == len(train_dataloader) - 1)
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
    
    if ddp:
        destroy_process_group()

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
        '--data_paths',
        type=str,
        nargs='+',
        required=True,
        help="Path to video dataset"
    )
    parser.add_argument(
        '--train_paths',
        type=str,
        nargs='+',
        required=True,
        help="Path to metadata file of the training video dataset"
    )
    parser.add_argument(
        '--eval_paths',
        type=str,
        nargs='+',
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
        '--per_device_train_batch_size',
        type=int,
        default=1,
        help='Per-device train batch size for running'
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=1,
        help='Per-device eval batch size for running'
    )
    args = parser.parse_args()
    os.makedirs(SAFETENSORS_PATH, exist_ok=True)
    # mp.set_start_method('spawn')

    train(args)
