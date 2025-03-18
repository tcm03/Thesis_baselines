import torch
import numpy as np
from PIL import Image
from transformers import BaseImageProcessor
from typing import List, Union, Tuple
from resource_logging import measure_resource_usage, MeasureResourceUsage
import logging
from constants import *

from decord import cpu, VideoReader

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(
    images: torch.Tensor, 
    image_processor: List[BaseImageProcessor]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    # images.shape: (4294, 360, 640, 3)
    # print(f'@tcm: In process_images(): images.shape={images.shape}')
    if isinstance(image_processor, list):
        processor_aux_list = image_processor
        new_images_aux_list = []
        for i, image in enumerate(images):
            # image.shape: (height, width, channels)
            # print(f'@tcm: In process_images(): frame {i}')
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image # PIL.Image
                if hasattr(processor_aux, "image_mean"):
                    try:
                        target_resolution = processor_aux.crop_size["height"]
                    except:
                        target_resolution = processor_aux.size["height"]
                    image_aux = expand2square(
                        image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                image_aux = processor_aux.preprocess(image_aux, return_tensors="pt")[
                    "pixel_values"
                ][0]
                # image_aux.shape: torch.Size([3, 384, 384])
                image_aux_list.append(image_aux)
            new_images_aux_list.append(image_aux_list) # torch.Tensor(C, H, W) new_images_aux_list[num_frames][num_processor]
        
        new_images_aux_list = [
            list(batch_image_aux) for batch_image_aux in zip(*new_images_aux_list)
        ] # torch.Tensor(C, H, W) new_images_aux_list[num_processor][num_frames]
        new_images_aux_list = [
            # torch.stack(image_aux).half() for image_aux in new_images_aux_list # @tcm: try full fp32 to match precision of vision encoders
            torch.stack(image_aux).float() for image_aux in new_images_aux_list
        ] # torch.Tensor(num_frames, C, H, W) new_images_aux_list[num_processor]
        return new_images_aux_list 
    else:
        image_aspect_ratio = "pad"
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

def process_video_frames(
    video_path: str,
    image_processors: List[BaseImageProcessor],
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
    # image_sizes = [vr[0].shape[:2]] # @tcm: I unwrap the shape tuple: [(x, y)] -> (x, y)
    image_sizes = vr[0].shape[:2]

    video = [[] for _ in range(len(image_processors))]
    for i in range(0, len(frame_indices), CHUNK_SIZE):
        sub_frame_indices = frame_indices[i:min(i+CHUNK_SIZE, len(frame_indices))]
        sub_videos = []
        for frame_index in sub_frame_indices:
            img = vr[frame_index].asnumpy()
            sub_videos.append(img)
        sub_videos = np.stack(sub_videos) # shape: (num_frames, height, width, channels)
        logging.info(f"Before process_images, sub_videos.dtype: {sub_videos.dtype}")
        sub_videos = process_images(sub_videos, image_processors)
        assert len(sub_videos) == len(video)
        logging.info(f"sub_videos[0] dtype: {sub_videos[0].dtype}")
        for j, sub_video in enumerate(sub_videos):
            video[j].append(sub_video)

    for i in range(len(video)):
        video[i] = torch.cat(video[i], dim=0)

    return video, image_sizes

def extract_filename(file_path: str) -> str:
    return file_path.split('/')[-1].split('.')[0]