import torch
from torch.utils.data import Dataset
from typing import List
import os
from mm_datautils import process_video_frames
from transformers import BaseImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from resource_logging import measure_resource_usage, MeasureResourceUsage
import logging
import decord

class EngagementDataset(Dataset):
    
    def __init__(
        self,   
        folder_paths: List[str],
        image_processors: List[BaseImageProcessor],
    ) -> None:
        self.file_paths = []
        self.image_processors = image_processors

        for folder_path in folder_paths:
            file_names = os.listdir(folder_path)
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                self.file_paths.append((file_path, file_name))
                
                # temporarily filter out long videos to handle OOM issues
                # vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)
                # duration = len(vr) / vr.get_avg_fps()
                # if duration <= 3600:
                #     self.file_paths.append((file_path, file_name))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video, image_size = process_video_frames(self.file_paths[idx][0], self.image_processors)
        return video, image_size, self.file_paths[idx][1]

def collate_fn(batch):
    """
    batch: list of samples from EngagementDataset.__getitem__()
    """
    assert isinstance(batch, list)
    assert isinstance(batch[0], tuple)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_sizes = [vid_data[1] for vid_data in batch]
    batch_videos = [video for video, _, _ in batch] # ignore image_size and file_name
    # batch_videos = [[video.to(device) for video in videos] for videos in zip(*batch_videos)]
    tmp_batch_videos = []
    for i, videos in enumerate(zip(*batch_videos)):
        # print(f'@tcm: processor {i}')
        tmp = []
        for j, video in enumerate(videos):
            # print(f'@tcm: video {j} shape: {video.shape}')
            # video = video.to(device) # try using data parallel w/o manual moving
            tmp.append(video)
        tmp_batch_videos.append(tmp)
    batch_videos = tmp_batch_videos
    return batch_videos, image_sizes, (file_name for _, _, file_name in batch)
