import torch
import torch.nn as nn
import torch.nn.init as init
from vision_encoders.builder import build_vision_tower_aux_list
from vision_sampler import VisionTokenSampler
from transformers import Qwen2Config
from typing import Optional, List, Tuple
import json
import math
from transformers import BaseImageProcessor
from resource_logging import *
import torch.nn.functional as F
import logging

def unmask_attention_mask(mask, original_size):
    original_w, original_h = original_size
    cur_h, cur_w = mask.shape[1:3]

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        if padding > 0:
            mask[:, :padding, :] = 0
            mask[:, -padding:, :] = 0
        return mask
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        if padding > 0:
            mask[:, :, :padding] = 0
            mask[:, :, -padding:] = 0
        return mask

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
        # if 0 in unpadded_tensor.shape:
        #     print(f"scale_factor: {scale_factor}, new_height: {new_height}, padding: {padding}, original_width: {original_width}, original_height: {original_height}")
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]
        # if 0 in unpadded_tensor.shape:
        #     print(f"scale_factor: {scale_factor}, new_width: {new_width}, padding: {padding}, original_width: {original_width}, original_height: {original_height}")

    return unpadded_tensor

class CambrianConfig(Qwen2Config):
    model_type = "cambrian_qwen"
    debug = "debug"

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json_file(cls, json_file_path):
        """Load a config from a json file."""
        with open(json_file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class CambrianMeta(nn.Module):

    def __init__(
        self, 
        config: CambrianConfig
    ) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.config: CambrianConfig = config

        vision_hidden_size = config.vision_hidden_size # 1024
        num_query_group = config.num_query_group # 1
        query_num_list = config.query_num_list # [144]
        connector_only = config.connector_only # true
        connector_depth = config.connector_depth # 3
        self.vision_tower_aux_list = nn.ModuleList(build_vision_tower_aux_list(
            config, delay_load=True
        ))
#        self.mm_projector = nn.Sequential(
#            nn.Linear(vision_hidden_size * num_query_group, config.hidden_size), # 3584
#            nn.GELU(),
#            nn.Linear(config.hidden_size, config.hidden_size), # 3584
#        )

        image_token_len = config.image_token_len # 144
        vision_tower_aux_token_len_list = (
            self.config.mm_vision_tower_aux_token_len_list
        ) # (576, 576)
        cross_att_token_len_list = [
            int(vision_tower_aux_token_len**0.5) // int(image_token_len**0.5)
            for vision_tower_aux_token_len in vision_tower_aux_token_len_list
        ]

        for aux_i, vision_tower_aux in enumerate(self.vision_tower_aux_list):
            setattr(
                self,
                "mm_projector_aux_{}".format(aux_i),
                nn.Sequential(
                    nn.Linear(vision_tower_aux.hidden_size, vision_hidden_size),
                    nn.GELU(),
                    nn.Linear(vision_hidden_size, vision_hidden_size),
                    nn.LayerNorm(vision_hidden_size),
                ),
            )
            

        for query_group_i in range(num_query_group):
            cross_att_token_len_list = [
                int(vision_tower_aux_token_len**0.5)
                // int(query_num_list[query_group_i] ** 0.5)
                for vision_tower_aux_token_len in vision_tower_aux_token_len_list
            ]
            setattr(
                self,
                "vision_sampler_{}".format(query_group_i),
                VisionTokenSampler(
                    vision_hidden_size,
                    vision_hidden_size,
                    [vision_hidden_size] * len(self.vision_tower_aux_list),
                    cross_att_token_len_list,
                    vision_hidden_size,
                    connector_depth,
                ),
            )

        if not connector_only:
            num_of_vision_sampler_layers = (
                config.num_of_vision_sampler_layers
            ) = config.num_of_vision_sampler_layers
            config.start_of_vision_sampler_layers = (
                config.start_of_vision_sampler_layers
            )
            config.stride_of_vision_sampler_layers = (
                config.stride_of_vision_sampler_layers
            )
            cross_att_token_len_list = [
                int(vision_tower_aux_token_len**0.5)
                // int(image_token_len**0.5)
                for vision_tower_aux_token_len in vision_tower_aux_token_len_list
            ]
            self.vision_sampler_layers = nn.ModuleList(
                [
                    VisionTokenSampler(
                        config.hidden_size,
                        vision_hidden_size,
                        [vision_hidden_size] * len(self.vision_tower_aux_list),
                        cross_att_token_len_list,
                        vision_hidden_size,
                        1,
                    )
                    for layer_idx in range(0, num_of_vision_sampler_layers)
                ]
            )

        self.vision_query = nn.Parameter(
            torch.randn((num_query_group, vision_hidden_size), dtype=self.dtype)
        )

#        self.image_newline = nn.Parameter(
#            torch.empty(config.hidden_size, dtype=self.dtype)
#        )

#        self.frame_pos = torch.stack(
#            [
#                1
#                / torch.pow(
#                    torch.tensor(10000),
#                    torch.tensor(2 * (hid_j // 2) / config.hidden_size),
#                )
#                for hid_j in range(config.hidden_size)
#            ]
#        )

        self.pad_num_frames = 60
        self.num_engagement_labels = 3
        # self.last_fc = nn.Linear(self.pad_num_frames * image_token_len * vision_hidden_size, self.num_engagement_labels)
        self.fc_1 = nn.Linear(self.pad_num_frames * image_token_len * vision_hidden_size, 128)
        self.bn_1 = nn.BatchNorm1d(128)
        self.fc_2 = nn.Linear(128, 32)
        self.bn_2 = nn.BatchNorm1d(32)
        self.fc_3 = nn.Linear(32, self.num_engagement_labels)
        self.bn_3 = nn.BatchNorm1d(self.num_engagement_labels)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc_1.weight)
        if self.fc_1.bias is not None:
            init.zeros_(self.fc_1.bias)
        init.xavier_uniform_(self.fc_2.weight)
        if self.fc_2.bias is not None:
            init.zeros_(self.fc_2.bias)
        init.xavier_uniform_(self.fc_3.weight)
        if self.fc_3.bias is not None:
            init.zeros_(self.fc_3.bias)
        # Initialize batch_norm layer
        init.ones_(self.bn_1.weight)
        init.zeros_(self.bn_1.bias)
        init.ones_(self.bn_2.weight)
        init.zeros_(self.bn_2.bias)
        init.ones_(self.bn_3.weight)
        init.zeros_(self.bn_3.bias)

    def get_frame_pos(self, time_range):
        frame_pos = self.frame_pos.reshape(1, -1) * time_range.reshape(-1, 1).to(
            self.frame_pos.device
        )
        frame_pos[:, 0::2] = torch.sin(frame_pos[:, 0::2])
        frame_pos[:, 1::2] = torch.cos(frame_pos[:, 0::2])
        frame_pos = frame_pos.unsqueeze(1)
        return frame_pos
    
    def rearrange_vision_tower_features_inference(
        self, vision_tower_aux_feature_list, query_side_len, image_sizes, unpad=False
    ):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        # logging.info(f"In rearrange_vision_tower_features_inference, bs={bs}, len(image_sizes)={len(image_sizes)}, image_sizes[0]={image_sizes[0]}")
        for vision_tower_aux_feature in vision_tower_aux_feature_list:
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1] ** 0.5)
            assert (aux_height // query_side_len) * query_side_len == aux_height

            reduce_factor = aux_height // query_side_len

            vision_tower_aux_feature_rearranged = []
            vision_tower_aux_attention_masks_rearranged = []
            for batch_i in range(bs):
                image_size = image_sizes[batch_i]
                cur_vision_tower_aux_feature = vision_tower_aux_feature[batch_i]

                cur_vision_tower_aux_attention_masks_rearranged = torch.ones(
                    (1, aux_height, aux_width),
                    dtype=torch.bool,
                    device=cur_vision_tower_aux_feature.device,
                )
                cur_vision_tower_aux_feature_rearranged = (
                    cur_vision_tower_aux_feature.view(
                        1,
                        query_side_len,
                        reduce_factor,
                        query_side_len,
                        reduce_factor,
                        -1,
                    )
                )
                cur_vision_tower_aux_feature_rearranged = (
                    cur_vision_tower_aux_feature_rearranged.permute(
                        0, 1, 3, 2, 4, 5
                    ).contiguous()
                )
                if unpad:
                    cur_vision_tower_aux_feature_rearranged = unpad_image(
                        cur_vision_tower_aux_feature_rearranged, image_size
                    )
                cur_vision_tower_aux_feature_rearranged = (
                    cur_vision_tower_aux_feature_rearranged.flatten(0, 2).flatten(1, 2)
                )  # query_side_len*query_side_len X reduce_factor*reduce_factor X C

                cur_vision_tower_aux_attention_masks_rearranged = unmask_attention_mask(
                    cur_vision_tower_aux_attention_masks_rearranged, image_size
                )
                cur_vision_tower_aux_attention_masks_rearranged = (
                    cur_vision_tower_aux_attention_masks_rearranged.view(
                        1, query_side_len, reduce_factor, query_side_len, reduce_factor
                    )
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                )
                if unpad:
                    cur_vision_tower_aux_attention_masks_rearranged = unpad_image(
                        cur_vision_tower_aux_attention_masks_rearranged, image_size
                    )
                cur_vision_tower_aux_attention_masks_rearranged = (
                    cur_vision_tower_aux_attention_masks_rearranged.flatten(
                        0, 2
                    ).flatten(1, 2)
                )

                cur_vision_tower_aux_attention_masks_rearranged[
                    cur_vision_tower_aux_attention_masks_rearranged.sum(-1) == 0
                ] = True

                vision_tower_aux_feature_rearranged.append(
                    cur_vision_tower_aux_feature_rearranged
                )
                vision_tower_aux_attention_masks_rearranged.append(
                    cur_vision_tower_aux_attention_masks_rearranged
                )

            vision_tower_aux_feature_rearranged = torch.cat(
                vision_tower_aux_feature_rearranged, 0
            )
            vision_tower_aux_attention_masks_rearranged = torch.cat(
                vision_tower_aux_attention_masks_rearranged, 0
            )

            vision_tower_aux_feature_rearranged_list.append(
                vision_tower_aux_feature_rearranged
            )
            vision_tower_aux_attention_masks_rearranged_list.append(
                vision_tower_aux_attention_masks_rearranged
            )

        return (
            vision_tower_aux_feature_rearranged_list,
            vision_tower_aux_attention_masks_rearranged_list,
        )
    
    def encode_images(self, image_aux_list, encode_type=None):
        vision_tower_aux_list = self.vision_tower_aux_list
        image_aux_features_list = []
        chunk_size = 64
        if encode_type == "dino":
            # print(f'@tcm: In CambrianMeta.encode_images(): dinov2')
            image_aux = image_aux_list[-1] # concatenated batch videos tensor for DINOv2: [# frames, 3, 378, 378]
            vision_tower_aux = vision_tower_aux_list[-1]
            if image_aux.shape[0] > chunk_size:
                image_aux_features_chunks = []
                for start_idx in range(0, image_aux.shape[0], chunk_size):
                    # print(f'@tcm: In CambrianMeta.encode_images(): dinov2 chunk start_idx={start_idx}')
                    end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                    chunk = image_aux[start_idx:end_idx]
                    image_aux_features_chunk = vision_tower_aux(chunk) # embedded shape: [# frames, 576, 1536]
                    image_aux_features_chunks.append(image_aux_features_chunk)
                image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
            else:
                # print(f'@tcm: In CambrianMeta.encode_images(): image_aux shape: {image_aux.shape}')
                image_aux_features = vision_tower_aux(image_aux)
            return image_aux_features
        elif encode_type == "siglip":
            # print(f'@tcm: In CambrianMeta.encode_images(): siglip')
            image_aux = image_aux_list[0]
            vision_tower_aux = vision_tower_aux_list[0]
            if image_aux.shape[0] > chunk_size:
                image_aux_features_chunks = []
                for start_idx in range(0, image_aux.shape[0], chunk_size):
                    # print(f'@tcm: In CambrianMeta.encode_images(): siglip chunk start_idx={start_idx}')
                    end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                    chunk = image_aux[start_idx:end_idx]
                    image_aux_features_chunk = vision_tower_aux(chunk)
                    image_aux_features_chunks.append(image_aux_features_chunk)
                image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
            else:
                image_aux_features = vision_tower_aux(image_aux)
            return image_aux_features
        else:
            # print(f'@tcm: In CambrianMeta.encode_images(): both encode_type')
            for image_aux, vision_tower_aux in zip(
                image_aux_list, vision_tower_aux_list
            ):
                if image_aux.shape[0] > chunk_size:
                    image_aux_features_chunks = []
                    for start_idx in range(0, image_aux.shape[0], chunk_size):
                        # print(f'@tcm: In CambrianMeta.encode_images(): both encode_type chunk start_idx={start_idx}')
                        end_idx = min(start_idx + chunk_size, image_aux.shape[0])
                        chunk = image_aux[start_idx:end_idx]
                        image_aux_features_chunk = vision_tower_aux(chunk)
                        image_aux_features_chunks.append(image_aux_features_chunk)
                    image_aux_features = torch.cat(image_aux_features_chunks, dim=0)
                else:
                    image_aux_features = vision_tower_aux(image_aux)
                image_aux_features_list.append(image_aux_features)
            return image_aux_features_list

    def select_frame(
        self,
        feature_list,
        split_sizes,
        new_image_aux_list,
        image_sizes,
        window_size=16,
        threshold=0.83,
    ):
        dino_features_batch = torch.split(feature_list, split_sizes, dim=0)
        new_image_aux_batch_0 = torch.split(new_image_aux_list[0], split_sizes, dim=0)
        new_image_aux_batch_1 = torch.split(new_image_aux_list[1], split_sizes, dim=0)
        new_split_sizes = []
        selected_frames_all_0 = []
        selected_frames_all_1 = []
        selected_frames_feature_all = []
        selected_frame_indices_all = []
        for i_batch, frame_features in enumerate(dino_features_batch):
            # print(f'@tcm: In CambrianMeta.select_frame(): dino features batch {i_batch}')
            if isinstance(frame_features, torch.Tensor):
                # print(f'@tcm: In CambrianMeta.select_frame(): frame_features shape: {frame_features.shape}')
                pass
            # original_width, original_height = image_sizes[i_batch]
            if getattr(self.config, "highres", False):
                token_per_frame = self.config.lowres_token ** 2
            else:
                token_per_frame = self.config.image_token_len

            max_num_frames = max(
                1,
                (
                    self.config.tokenizer_model_max_length
                    - getattr(self.config, "inference_max_length", 16)
                )
                // token_per_frame,
            )
            if len(frame_features) < max_num_frames:
                selected_frames_all_0.append(new_image_aux_batch_0[i_batch])
                selected_frames_all_1.append(new_image_aux_batch_1[i_batch])
                selected_frames_feature_all.append(frame_features)
                new_split_sizes.append(len(frame_features))
                selected_frame_indices_all.append(torch.arange(len(frame_features)))
                continue

            num_segments = len(frame_features) // window_size
            if num_segments == 0:
                query_feature = frame_features.flatten(1, 2)
                query_feature = query_feature / torch.norm(
                    (query_feature), dim=1, keepdim=True
                )
                similarities = torch.mean(query_feature @ query_feature.T, dim=1)
                similarities[len(frame_features) // 2] = 0
                indices = torch.where(similarities < threshold)[0]
                selected_frame_indices_all.append(indices)
                selected_frames_all_0.append(new_image_aux_batch_0[i_batch][indices])
                selected_frames_all_1.append(new_image_aux_batch_1[i_batch][indices])
                selected_frames_feature_all.append(frame_features[indices])
                new_split_sizes.append(len(indices))
                continue
            segments_frames_0 = []
            segments_frames_1 = []
            segments_features = []
            for start_idx in range(0, len(frame_features), window_size):
                end_idx = min(start_idx + window_size, len(frame_features))
                segments_frames_0.append(
                    new_image_aux_batch_0[i_batch][start_idx:end_idx]
                )
                segments_frames_1.append(
                    new_image_aux_batch_1[i_batch][start_idx:end_idx]
                )
                segments_features.append(frame_features[start_idx:end_idx])
            selected_frames_0 = []
            selected_frames_1 = []
            selected_features = []
            selected_frame_indices = []
            for i, segment in enumerate(segments_features):
                query_feature = segment.flatten(1, 2)
                query_feature = query_feature / torch.norm(
                    (query_feature), dim=1, keepdim=True
                )
                similarities = torch.mean(query_feature @ query_feature.T, dim=1)
                similarities[len(segment) // 2] = 0
                indices = torch.where(similarities < threshold)[0]
                selected_frames_0.append(segments_frames_0[i][indices])
                selected_frames_1.append(segments_frames_1[i][indices])
                selected_features.append(segment[indices])
                selected_frame_indices.extend(indices + i * window_size)
            selected_frames_0 = torch.cat(selected_frames_0, dim=0)
            selected_frames_1 = torch.cat(selected_frames_1, dim=0)
            selected_features = torch.cat(selected_features, dim=0)
            selected_frame_indices = torch.tensor(selected_frame_indices)
            # ablation
            max_num_frames = 400  # in case of OOM
            if len(selected_frames_0) > max_num_frames:
                interval = len(selected_frames_0) / float(max_num_frames)
                indices = [int(interval * i) for i in range(max_num_frames)]
                new_split_sizes.append(len(indices))
                selected_frames_all_0.append(selected_frames_0[indices])
                selected_frames_all_1.append(selected_frames_1[indices])
                selected_frames_feature_all.append(selected_features[indices])
                selected_frame_indices = selected_frame_indices[indices]
            else:
                new_split_sizes.append(len(selected_frames_0))
                selected_frames_all_0.append(selected_frames_0)
                selected_frames_all_1.append(selected_frames_1)
                selected_frames_feature_all.append(selected_features)
            selected_frame_indices_all.append(selected_frame_indices)
        selected_frames_all_0 = torch.cat(selected_frames_all_0, dim=0)
        selected_frames_all_1 = torch.cat(selected_frames_all_1, dim=0)
        selected_frames_feature_all = torch.cat(selected_frames_feature_all, dim=0)
        return (
            selected_frames_feature_all,
            new_split_sizes,
            [selected_frames_all_0, selected_frames_all_1],
            selected_frame_indices_all,
        )
    

    def prepare_mm_features(
        self,
        # input_ids,
        images: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]], # @tcm: List of frame sizes of videos in a batch (frames in a video have same size)
    ):
        # logging.info(f'In the beginning: image_sizes: {image_sizes}')
        vision_tower_aux_list = self.vision_tower_aux_list
        image_aux_list = images

#        for i, image in enumerate(image_aux_list[0]):
#            debug_tensor(f'image_aux_list[0][{i}]', image_aux_list[0][i])        
        
        split_sizes_ori = [
            1 if image.ndim == 3 else image.shape[0] for image in image_aux_list[0]
        ] # number of frames of each video in the bach (len = batch size)
        new_image_aux_list = []
        for image_aux in image_aux_list:
            if type(image_aux) is list:
                # image_aux = [
                #     x.unsqueeze(0) if x.ndim == 3 else x for x in image_aux
                # ]
                tmp_image_aux = []
                for x in image_aux:
                    assert x.ndim == 3 or x.ndim == 4, 'Each video tensor should have 3 (#frames=1) or 4 dimensions'
                    # print(f'@tcm: In CambrianMeta.prepare_mm_features(): x.shape={x.shape}')
                    if x.ndim == 3:
                        # add num_frames dimension
                        x = x.unsqueeze(0)
                    # if x.shape[0] == 1:
                    #     x = x.squeeze(0)
                    tmp_image_aux.append(x)
                image_aux = tmp_image_aux
                
            # @tcm: image_aux is a list of tensors of shape (# frames, C, H, W)
            # @tcm: We concatenate all videos along the num frames dimension
            concat_image_aux = torch.cat([image for image in image_aux], dim=0) # (sum # frames, C, H, W)
            new_image_aux_list.append(concat_image_aux) # for each vision encoder

        image_aux_features_dino = self.encode_images(
            new_image_aux_list, encode_type="dino"
        ) # # image_aux_features_dino.shape: [sum # frames, 576, 1536]

        (
            image_aux_features_dino,
            split_sizes,
            new_image_aux_list,
            selected_frame_indices_all,
        ) = self.select_frame(
            image_aux_features_dino,
            split_sizes_ori,
            new_image_aux_list,
            image_sizes,
            threshold=getattr(self.config, "dino_threshold", 0.83),
        )
        
        image_aux_features_siglip = self.encode_images(
            new_image_aux_list, encode_type="siglip"
        ) # image_aux_features_siglip.shape: [sum # frames, 576, 1152]

        image_aux_features_list = [
            image_aux_features_siglip,
            image_aux_features_dino,
        ]
        
        bs = image_aux_features_list[0].shape[0] # sum # frames
        dtype = new_image_aux_list[0].dtype

        frame_sizes = []
        # logging.info(f'image_sizes: {image_sizes}')
        logging.info(f'In prepare_mm_features: split_sizes: {split_sizes}')
        for i in range(len(image_sizes)):
            for j in range(split_sizes[i]):
                frame_sizes.append(image_sizes[i])
        image_sizes = frame_sizes # [(w, h), ..., (w, h)] (len = sum # frames)
        
        image_token_len = self.config.image_token_len
        query_num_list = self.config.query_num_list

        final_height = final_width = int(image_token_len**0.5)

        final_image_features_list = []
        final_image_features_down_list = []

        # only needed for sva
        vision_tower_aux_feature_list_final = None
        vision_tower_aux_attention_masks_list_final = None
        global_context_feature_final = None

        if self.config.mm_projector_type == "sva":
            vision_tower_aux_feature_list = []
            vision_tower_aux_attention_masks_list = []
            # get vision tokens from each vision tower
            for aux_i in range(len(vision_tower_aux_list)):
                image_aux_features = image_aux_features_list[aux_i]
                mm_proj = getattr(self, "mm_projector_aux_{}".format(aux_i))
                image_aux_features = mm_proj(image_aux_features).to(dtype)
                # [sum # frames, 576, 1152] -> [sum # frames, 576, 1024]
                # [sum # frames, 576, 1536] -> [sum # frames, 576, 1024]
                if aux_i == 0:
                    # global_context_feature.shape: [sum # frames, 1, 1, 1024]
                    global_context_feature = image_aux_features.mean(1).view(
                        bs, 1, 1, -1
                    )

                vision_tower_aux_feature_list.append(image_aux_features)

            
            input_mix_res = True
            input_high_res = True
            # perform vision sampling for each query group
            for query_group_i, query_num in enumerate(query_num_list):
                query_features_i = (
                    self
                    .vision_query[query_group_i, :]
                    .view(1, 1, 1, -1)
                    .expand(bs, query_num, -1, -1)
                )
                # query_features_i.shape: [sum # frames, 144, 1, 1024]
                global_context_feature_i = global_context_feature.expand(
                    -1, query_num, 1, -1
                ).flatten(0, 1)
                # global_context_feature_i.shape: [sum # frames * 144, 1, 1024]
                query_side_len = int(query_num**0.5) # 12
                (
                    vision_tower_aux_feature_list_i,
                    vision_tower_aux_attention_masks_list_i,
                ) = self.rearrange_vision_tower_features_inference(
                    vision_tower_aux_feature_list, query_side_len, image_sizes
                )

                query_features_i = getattr(
                    self, "vision_sampler_{}".format(query_group_i)
                )(
                    query_features_i.flatten(0, 1),
                    global_context_feature_i,
                    *vision_tower_aux_feature_list_i,
                    *vision_tower_aux_attention_masks_list_i,
                )
                query_features_i = query_features_i.view(bs, query_num, -1)
                # query_features_i.shape: [sum # frames, 144, 1024]

                if split_sizes is not None:
                    try:
#                        if "llama" in self.config.model_type:
#                            text_len = torch.where(input_ids[0] == 128002)[-1][0]
#                        else:
#                            text_len = torch.where(input_ids[0] == 151643)[-1][0]
                        text_len = 0 # @tcm: I don't consider text in this non-multimodal baseline
                    except:
                        # text_len = len(input_ids[0])
                        text_len = 0 # @tcm: I don't consider text in this non-multimodal baseline
                    max_visual_len = (
                        self.config.tokenizer_model_max_length
                        - text_len
                        - getattr(self.config, "inference_max_length", 16)
                    )
                    max_num_frames = max(
                        1,
                        math.floor(max_visual_len // (final_height * final_width)),
                    )
                    max_num_frames_low = max(
                        1,
                        math.floor(
                            max_visual_len
                            // (self.config.lowres_token ** 2)
                        ),
                    )
                    if split_sizes[0] < max_num_frames:
                        input_mix_res = False
                    elif split_sizes[0] > max_num_frames_low:
                        input_mix_res = False
                        input_high_res = False

                # input_mix_res = False  # ablation

                if (getattr(self.config, "highres", False)) and input_mix_res:
                    _query_features_i = (
                        query_features_i.permute(0, 2, 1)
                        .contiguous()
                        .view(bs, -1, query_side_len, query_side_len)
                    )
                    _query_features_i = F.interpolate(
                        _query_features_i.float(),
                        size=(
                            self.config.lowres_token,
                            self.config.lowres_token,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=query_features_i.dtype)
                    _query_features_i = (
                        _query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                    )
                    final_image_features_down_list.append(_query_features_i)

                # interpolate to the final target size
                if query_side_len != final_height:
                    query_features_i = (
                        query_features_i.permute(0, 2, 1)
                        .contiguous()
                        .view(bs, -1, query_side_len, query_side_len)
                    )
                    if input_high_res:
                        query_features_i = F.interpolate(
                            query_features_i.float(),
                            size=(final_height, final_width),
                            mode="bilinear",
                            align_corners=False,
                        ).to(dtype=query_features_i.dtype)
                    else:
                        query_features_i = F.interpolate(
                            query_features_i.float(),
                            size=(8, 8),
                            mode="bilinear",
                            align_corners=False,
                        ).to(dtype=query_features_i.dtype)
                    query_features_i = (
                        query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                    )
                final_image_features_list.append(query_features_i)
        
        image_features = torch.cat(final_image_features_list, -1) # image_features.shape: [sum # frames, 144, 1024]

        final_image_features_list = []
        start_frame_idx = 0
        for batch_i, cur_num_frames in enumerate(split_sizes):
            cur_frames_features = image_features[start_frame_idx:start_frame_idx + cur_num_frames]
            start_frame_idx += cur_num_frames
            # if first dim (num frames) less than 60, pad it to 60 with zeros
            if cur_frames_features.shape[0] < self.pad_num_frames:
                cur_seq_len = cur_frames_features.shape[1]
                cur_hidden_size = cur_frames_features.shape[2]
                pad = torch.zeros((self.pad_num_frames - cur_frames_features.shape[0], cur_seq_len, cur_hidden_size), dtype=dtype, device=self.vision_query.device)
#                logging.info(f"debug: cur_frames_features.device: {cur_frames_features.device}")
#                logging.info(f"debug: pad.device: {pad.device}")
                cur_frames_features = torch.cat([cur_frames_features, pad], dim=0)
            elif cur_frames_features.shape[0] > self.pad_num_frames:
                cur_frames_features = cur_frames_features[:self.pad_num_frames]
            cur_frames_features = cur_frames_features.flatten(0, 2)
            final_image_features_list.append(cur_frames_features)
        final_image_features_batch = torch.stack(final_image_features_list, dim=0) # [batch size, 60 * 144 * 1024]

        return final_image_features_batch


        # image_features = self.mm_projector(image_features).to(dtype) # image_features.shape: [sum # frames, 144, 3072] 


        # if (getattr(self.config, "highres", False)) and input_mix_res:
        #     image_features_down = torch.cat(final_image_features_down_list, -1)
        #     image_features_down = (
        #         self.mm_projector(image_features_down).to(dtype)
        #     )

        # image_features = image_features.view(bs, final_height, final_width, -1) # image_features.shape: [sum # frames, 12, 12, 3072]

        # if (getattr(self.config, "highres", False)) and input_mix_res:
        #     image_features_down = image_features_down.view(
        #         bs,
        #         self.config.lowres_token,
        #         self.config.lowres_token,
        #         -1,
        #     )
        # image_features_unpadded = []
        # image_features_downsample = []
        # final_size = []
        # if self.config.mm_projector_type == "sva":
        #     (
        #         vision_tower_aux_feature_list_final,
        #         vision_tower_aux_attention_masks_list_final,
        #     ) = self.rearrange_vision_tower_features_inference(
        #         vision_tower_aux_feature_list, final_height, image_sizes, unpad=True
        #     )
        #     global_context_feature_final = []
        # for batch_i in range(bs):
        #     cur_image_feature = image_features[batch_i] # cur_image_feature.shape: [12, 12, 3072]
        #     image_size = image_sizes[batch_i] # image_size: (360, 640)

        #     cur_image_feature = unpad_image(
        #         cur_image_feature.unsqueeze(0), image_size
        #     ) # # cur_image_feature.shape: [1, 12, 6, 3072] (unpad height)

        #     cur_h, cur_w = cur_image_feature.shape[1:3]
        #     try:  # fix bug for some invalid image
        #         cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
        #         final_size.append((cur_h, cur_w))
        #     except:
        #         # print(f"invalid after unpad {image_features[batch_i].shape}, {image_sizes[batch_i]}", flush=True)
        #         cur_image_feature = image_features[batch_i].unsqueeze(0)
        #         image_size = image_sizes[batch_i]
        #         cur_h, cur_w = cur_image_feature.shape[1:3]
        #         cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
        #         final_size.append((cur_h, cur_w))

        #     if (getattr(self.config, "highres", False)) and input_mix_res:
        #         cur_image_feature_down = unpad_image(
        #             image_features_down[batch_i].unsqueeze(0),
        #             (
        #                 int(
        #                     image_size[0]
        #                     / (
        #                         image_token_len**0.5
        #                         / self.config.lowres_token
        #                     )
        #                 ),
        #                 int(
        #                     image_size[1]
        #                     / (
        #                         image_token_len**0.5
        #                         / self.config.lowres_token
        #                     )
        #                 ),
        #             ),
        #         )
        #         _cur_h, _cur_w = cur_image_feature_down.shape[1:3]

        #         try:  # fix bug for some invalid image
        #             cur_image_feature_down = cur_image_feature_down.view(
        #                 1, _cur_h, _cur_w, -1
        #             )
        #         except:
        #             print("invalid after unpad", flush=True)
        #             cur_image_feature_down = image_features_down[batch_i].unsqueeze(
        #                 0
        #             )
        #             _cur_h, _cur_w = cur_image_feature_down.shape[1:3]
        #             cur_image_feature_down = cur_image_feature_down.view(
        #                 1, _cur_h, _cur_w, -1
        #             )

        #         cur_image_feature_down = torch.cat(
        #             (
        #                 cur_image_feature_down,
        #                 self.image_newline.view(1, 1, 1, -1)
        #                 .expand(1, _cur_h, 1, -1)
        #                 .to(cur_image_feature_down.device),
        #             ),
        #             dim=2,
        #         ).flatten(1, 2)

        #         if split_sizes is None and getattr(self.config, "frame_pos", False):
        #             frame_pos = (
        #                 self
        #                 .get_frame_pos(torch.arange(1))
        #                 .to(cur_image_feature_down.device)
        #                 .to(cur_image_feature_down.dtype)
        #             )
        #             cur_image_feature_down += frame_pos

        #         image_features_downsample.append(cur_image_feature_down.squeeze(0))

        #     cur_image_feature = torch.cat(
        #         (
        #             cur_image_feature,
        #             self.image_newline.view(1, 1, 1, -1)
        #             .expand(1, cur_h, 1, -1)
        #             .to(cur_image_feature.device),
        #         ),
        #         dim=2,
        #     )

        #     if split_sizes is None and getattr(self.config, "frame_pos", False):
        #         frame_pos = (
        #             self
        #             .get_frame_pos(torch.arange(1))
        #             .to(cur_image_feature.device)
        #             .to(cur_image_feature.dtype)
        #         )
        #         cur_image_feature += frame_pos

        #     cur_image_feature = cur_image_feature.flatten(1, 2)
        #     image_features_unpadded.append(cur_image_feature.squeeze(0))

        #     if self.config.mm_projector_type == "sva":
        #         cur_global_context_feature = global_context_feature[batch_i].expand(
        #             cur_h * cur_w, 1, -1
        #         )
        #         global_context_feature_final.append(cur_global_context_feature)
        # if self.config.mm_projector_type == "sva":
        #     global_context_feature_final = torch.cat(
        #         global_context_feature_final, 0
        #     )

        # if (getattr(self.config, "highres", False)) and input_mix_res:
        #     image_features = image_features_downsample
        # else:
        #     image_features = image_features_unpadded

        # # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
        #     self.config, "mm_use_im_start_end", False
        # ):
        #     raise NotImplementedError

        # split_image_features_unpadded = None
        # frame_split_sizes = None

        # if split_sizes is not None:
        #     split_image_features = []
        #     split_image_features_unpadded = (
        #         []
        #         if (getattr(self.config, "highres", False)) and input_mix_res
        #         else None
        #     )
        #     start_idx = 0
        #     for split_batch_idx, split_size in enumerate(split_sizes):
        #         if isinstance(image_features[start_idx : start_idx + split_size], list):
        #             if getattr(self.config, "frame_pos", False):
        #                 frame_feature = torch.cat(
        #                     image_features[start_idx : start_idx + split_size], dim=0
        #                 ).reshape(split_size, -1, image_features[0].shape[-1])
        #                 frame_pos = (
        #                     self
        #                     .get_frame_pos(selected_frame_indices_all[split_batch_idx])
        #                     .to(frame_feature.device)
        #                     .to(frame_feature.dtype)
        #                 )
        #                 frame_feature += frame_pos
        #                 split_image_features.append(
        #                     frame_feature.reshape(-1, image_features[0].shape[-1])
        #                 )
        #             else:
        #                 split_image_features.append(
        #                     torch.cat(
        #                         image_features[start_idx : start_idx + split_size],
        #                         dim=0,
        #                     )
        #                 )
        #             if (getattr(self.config, "highres", False)) and input_mix_res:
        #                 if getattr(self.config, "frame_pos", False):
        #                     frame_feature = torch.cat(
        #                         image_features_unpadded[
        #                             start_idx : start_idx + split_size
        #                         ],
        #                         dim=0,
        #                     ).reshape(split_size, -1, image_features[0].shape[-1])
        #                     frame_pos = (
        #                         self
        #                         .get_frame_pos(
        #                             selected_frame_indices_all[split_batch_idx]
        #                         )
        #                         .to(frame_feature.device)
        #                         .to(frame_feature.dtype)
        #                     )
        #                     frame_feature += frame_pos
        #                     split_image_features_unpadded.append(
        #                         frame_feature.reshape(-1, image_features[0].shape[-1])
        #                     )
        #                 else:
        #                     split_image_features_unpadded.append(
        #                         torch.cat(
        #                             image_features_unpadded[
        #                                 start_idx : start_idx + split_size
        #                             ],
        #                             dim=0,
        #                         )
        #                     )
        #         else:
        #             if getattr(self.config, "frame_pos", False):
        #                 frame_feature = image_features[
        #                     start_idx : start_idx + split_size
        #                 ].reshape(split_size, -1, image_features[0].shape[-1])
        #                 frame_pos = (
        #                     self
        #                     .get_frame_pos(selected_frame_indices_all[split_batch_idx])
        #                     .to(frame_feature.device)
        #                     .to(frame_feature.dtype)
        #                 )
        #                 frame_feature += frame_pos
        #                 split_image_features.append(
        #                     frame_feature.reshape(-1, image_features[0].shape[-1])
        #                 )
        #             else:
        #                 split_image_features.append(
        #                     image_features[start_idx : start_idx + split_size]
        #                 )
        #             if (getattr(self.config, "highres", False)) and input_mix_res:
        #                 if getattr(self.config, "frame_pos", False):
        #                     frame_feature = image_features_unpadded[
        #                         start_idx : start_idx + split_size
        #                     ]
        #                     frame_pos = (
        #                         self
        #                         .get_frame_pos(
        #                             selected_frame_indices_all[split_batch_idx]
        #                         )
        #                         .to(frame_feature.device)
        #                         .to(frame_feature.dtype)
        #                     )
        #                     frame_feature += frame_pos
        #                     split_image_features_unpadded.append(
        #                         frame_feature.reshape(-1, image_features[0].shape[-1])
        #                     )
        #                 else:
        #                     split_image_features_unpadded.append(
        #                         image_features_unpadded[
        #                             start_idx : start_idx + split_size
        #                         ]
        #                     )
        #         start_idx += split_size
        #     image_features = split_image_features
        #     frame_split_sizes = split_sizes


        # return image_aux_features_list
    
    def forward(
        self,
        # input_ids,
        images: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]], # @tcm: List of frame sizes of videos in a batch (frames in a video have same size)
    ):
        
        image_features_batch = self.prepare_mm_features(
            # input_ids,
            images,
            image_sizes,
        )
        o1 = self.fc_1(image_features_batch)
        o1 = self.bn_1(o1)
        o2 = self.fc_2(o1)
        o2 = self.bn_2(o2)
        o3 = self.fc_3(o2)
        o3 = self.bn_3(o3)
        return o3
        
