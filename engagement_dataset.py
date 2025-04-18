import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import os
import json
from mm_datautils import process_video_frames
from utils import extract_filename
from transformers import BaseImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from resource_logging import measure_resource_usage, MeasureResourceUsage
import logging
import decord

class DatasetItem:

    def __init__(
        self, 
        file_name: str, 
        video_tensor: torch.Tensor, 
        image_size: Tuple[int, int], 
        label: int
    ) -> None:
        self.file_name = file_name
        self.video_tensor = video_tensor
        self.image_size = image_size
        self.label = label

class EngagementDataset(Dataset):
    
    def __init__(
        self,
        data_paths: List[str],
        json_paths: List[str],
        image_processors: List[BaseImageProcessor],
    ) -> None:
        assert len(data_paths) == len(json_paths), "data_paths and json_paths must have the same length"
        self.data: Tuple[str, str, int] = [] # [(filename, video_path, label)]
        self.image_processors: List[BaseImageProcessor] = image_processors

        for i in range(len(data_paths)):
            json_path = json_paths[i]
            data_path = data_paths[i]
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                assert isinstance(json_data, list), f"Json format must be a list, but got {type(json_data)}"
                for item in json_data:
                    video_path = os.path.join(data_path, item["video"])
                    label = int(item["label"])
                    filename = extract_filename(video_path)
                    self.data.append((filename, video_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video, image_size = process_video_frames(self.data[idx][1], self.image_processors)
        # video: List[torch.Size([60, 3, 384, 384]), torch.Size([60, 3, 378, 378])]
        filename = self.data[idx][0]
        label = torch.tensor([self.data[idx][2]], dtype=torch.long)
        return DatasetItem(filename, video, image_size, label)

def collate_fn(batch):
    """
    Args:
        batch: list of samples from EngagementDataset.__getitem__()
    Returns:
        filenames: tuple of file names of videos in the batch
        batch_videos: tuple of two elements
            0: tuple of video tensors for SigLIP encoder
            1: tuple of video tensors for DINOv2 encoder
        image_sizes: tuple of image sizes for videos in the batch
        labels: torch.Tensor of labels for videos in the batch
    """
    assert isinstance(batch, list) or isinstance(batch, tuple)
    assert all(isinstance(item, DatasetItem) for item in batch)

    image_sizes = tuple([item.image_size for item in batch])
    batch_videos = tuple([item.video_tensor for item in batch])
    filenames = tuple([item.file_name for item in batch])
    labels = torch.cat([item.label for item in batch], dim=0)
    tmp_batch_videos = []
    for i, videos in enumerate(zip(*batch_videos)):
        tmp = []
        for j, video in enumerate(videos):
            tmp.append(video)
        tmp_batch_videos.append(tmp)
    batch_videos = tmp_batch_videos
    return filenames, batch_videos, image_sizes, labels
