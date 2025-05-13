import sys
sys.path.append('.')
sys.path.append('..')

import json
import os
import copy
from PIL import Image, ImageSequence
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from typing import Dict, Sequence, List
from dataclasses import dataclass, field

from backbones.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from models.multimodal_prep import prepare_multimodal_data
from backbones.mm_datautils import preprocess, preprocess_multimodal


class LazySupervisedDataset(Dataset):

    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        # pyre-fixme[2]: Parameter must be annotated.
        data_args,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        for data_path in data_paths:
            list_data_dict.extend(json.load(open(data_path, "r")))

        self.tokenizer = tokenizer
        # pyre-fixme[4]: Attribute must be annotated.
        self.list_data_dict = list_data_dict
        # pyre-fixme[4]: Attribute must be annotated.
        self.data_args = data_args
        # pyre-fixme[4]: Attribute must be annotated.
        self.length = self._get_length()

    # # pyre-fixme[3]: Return type must be annotated.
    # def _get_length(self):
    #     """Calculates the number of samples in the .jsonl file."""
    #     with open(self.data_path, "r") as file:
    #         for i, _ in enumerate(file):
    #             pass
    #     return i + 1  # pyre-fixme
    def _get_length(self):
        """Calculates the total number of samples across all .jsonl files."""
        total_lines = 0
        for path in self.data_paths:
            with open(path, "r") as file:
                total_lines += sum(1 for _ in file)
        return total_lines

    def __len__(self) -> int:
        return len(self.list_data_dict)

    # pyre-fixme[3]: Return type must be annotated.
    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, "length_list") and hasattr(self, "modality_length_list"):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list  # pyre-fixme

        self.length_list = []
        self.modality_length_list = []
        for sample in self.list_data_dict:
            img_tokens = (
                self.data_args.image_token_len if self._has_image(sample) else 0
            )
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            self.length_list.append(cur_len + img_tokens)
            modality_len = cur_len if "image" in sample else -cur_len
            self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    def _has_image(self, sample: dict) -> bool:  # pyre-fixme
        if "image" in sample and not str(sample["image"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        if "video" in sample and not str(sample["video"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        return False

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        dat = sources
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = self._has_image(dat)
        if has_image:
            if "image" in dat:
                image_file = dat["image"]
                image_folders = self.data_args.image_folders
                image_folder = None
                for img_folder in image_folders:
                    if os.path.exists(os.path.join(img_folder, image_file)):
                        image_folder = img_folder
                        break
                assert image_folder is not None, f"Image folder not found for {image_file}"
                processor_aux_list = self.data_args.image_processor_aux_list
                try:
                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
                except:
                    print(
                        "Not exist: ",
                        os.path.join(image_folder, image_file),
                        flush=True,
                    )
                    return self.__getitem__(0)
                image_size = image.size
            else:
                video_file = dat["video"]
                processor_aux_list = self.data_args.image_processor_aux_list
                if video_file.endswith(".gif"):
                    image_folder = None
                    for img_folder in image_folders:
                        if os.path.exists(os.path.join(img_folder, "gifs", video_file)):
                            image_folder = img_folder
                            break
                    assert image_folder is not None, f"Image folder not found for {video_file}"
                    video_file = os.path.join(
                        image_folder, "gifs", video_file
                    )
                else:
                    image_folder = None
                    for img_folder in image_folders:
                        if os.path.exists(os.path.join(img_folder, video_file)):
                            image_folder = img_folder
                            break
                    assert image_folder is not None, f"Image folder not found for {video_file}"
                    video_file = os.path.join(image_folder, video_file)
                if os.path.exists(video_file):
                    try:
                        if video_file.endswith(".npy"):
                            image = np.load(video_file)
                            image_size = image[0].shape[:2]
                        elif video_file.endswith(".gif"):
                            video = Image.open(video_file)
                            image = []
                            for frame in ImageSequence.Iterator(video):
                                frame_copy = frame.copy()
                                image.append(frame_copy.convert("RGB"))
                            image_size = image[0].size
                        elif os.path.isdir(video_file):
                            files = [f for f in sorted(os.listdir(video_file))]
                            image = []
                            for file in files:
                                image.append(
                                    Image.open(os.path.join(video_file, file)).convert(
                                        "RGB"
                                    )
                                )
                            image_size = image[0].size
                        else:
                            vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                            sample_fps = round(
                                vr.get_avg_fps() / self.data_args.video_fps
                            )
                            frame_idx = [i for i in range(0, len(vr), sample_fps)]
                            image = vr.get_batch(frame_idx).asnumpy()
                            image_size = image[0].shape[:2]
                        if self.data_args.uniform_sample:
                            num_sample = 100
                            if len(image) > num_sample:
                                interval = len(image) / float(num_sample)
                                indices = [int(interval * i) for i in range(num_sample)]
                                image = [image[idx] for idx in indices]
                    except:
                        print("fail to load video: ", video_file, flush=True)
                        return self.__getitem__(0)
                else:
                    print("Not exist: ", video_file, flush=True)
                    return self.__getitem__(0)

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    # result.paste(pil_img, (0, 0))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    # result.paste(pil_img, (0, 0))
                    return result

            if self.data_args.image_aspect_ratio != "pad":
                raise NotImplementedError("Only pad is supported for now.")

            image_aux_list = []
            # @tcm: use image processors
            for processor_aux in processor_aux_list:
                image_aux = image
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                if not isinstance(image_aux, Image.Image):
                    frame_list = []
                    for frame in image_aux:
                        if not isinstance(frame, Image.Image):
                            frame = Image.fromarray(frame)
                        frame_aux = expand2square(
                            frame, tuple(int(x * 255) for x in processor_aux.image_mean)
                        ).resize((target_resolution, target_resolution))
                        frame_aux = processor_aux.preprocess(
                            frame_aux, return_tensors="pt"
                        )["pixel_values"][0]
                        frame_list.append(frame_aux)
                    image_aux = torch.stack(frame_list)
                else:
                    image_aux = expand2square(
                        image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                    image_aux = processor_aux.preprocess(
                        image_aux, return_tensors="pt"
                    )["pixel_values"][0]
                image_aux_list.append(image_aux)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)  # pyre-fixme
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )
        if (data_dict["labels"] != IGNORE_INDEX).sum() == 0:
            return self.__getitem__(0)
        # image exist in the data
        if has_image:
            data_dict["image_aux_list"] = image_aux_list  # pyre-fixme
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            image_list = []
            for processor_aux in processor_aux_list:
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                image_list.append(
                    torch.zeros(
                        3,
                        target_resolution,
                        target_resolution,
                    )
                )
            data_dict["image_aux_list"] = image_list
            image_size = (crop_size, crop_size)
        data_dict["image_size"] = image_size  # pyre-fixme
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int
    image_aux_token_len_list: list  # pyre-fixme
    image_position: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:  # pyre-fixme

        image_token_len = self.image_token_len
        image_aux_token_len_list = self.image_aux_token_len_list
        image_position = self.image_position

        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side

        # print_rank0("Pad token id is", self.tokenizer.pad_token_id)

        if padding_side == "left":
            input_ids = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t,
                        (max_length - t.shape[0], 0),
                        "constant",
                        self.tokenizer.pad_token_id,
                    )
                )
                for t in input_ids
            ]
            labels = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t, (max_length - t.shape[0], 0), "constant", IGNORE_INDEX
                    )
                )
                for t in labels
            ]
        else:
            input_ids = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t,
                        (0, max_length - t.shape[0]),
                        "constant",
                        self.tokenizer.pad_token_id,
                    ) # @tcm: t is 1-dim tensor, pad at the end of t with pad_token_id
                )
                for t in input_ids
            ]
            labels = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t, (0, max_length - t.shape[0]), "constant", IGNORE_INDEX
                    ) # @tcm: t is 1-dim tensor, pad at the end of t with IGNORE_INDEX
                )
                for t in labels
            ]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # pyre-fixme
        # @tcm: insert dummy image to tokenized text input_ids if there is none
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_ids_tmp = input_ids[i].clone()
                cur_input_ids_tmp[image_position + 1 :] = input_ids[
                    i, image_position:-1
                ]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp

                cur_labels_tmp = labels[i].clone()
                cur_labels_tmp[image_position + 1 :] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp

                cur_attention_mask_tmp = attention_mask[i].clone()
                cur_attention_mask_tmp[image_position + 1 :] = attention_mask[
                    i, image_position:-1
                ]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp
        image_sizes = [instance["image_size"] for instance in instances]
        (
            new_input_ids,
            new_labels,
            new_attention_mask,
            new_position_ids,
            im_aux_attention_masks_list,
        ) = prepare_multimodal_data(
            input_ids,
            labels,
            attention_mask,
            image_sizes,
            image_token_len,
            image_aux_token_len_list,
            max_length,
        )
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            image_aux_attention_masks_list=im_aux_attention_masks_list,
        )
        batch["image_sizes"] = image_sizes
        if "image_aux_list" in instances[0]:
            image_aux_list = [instance["image_aux_list"] for instance in instances]
            image_aux_list = [
                list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)
            ]
            if all(
                x is not None and x.shape == image_aux_list[0][0].shape
                for x in image_aux_list[0]
            ):
                batch["images"] = [
                    torch.stack(image_aux) for image_aux in image_aux_list
                ]
            else:
                batch["images"] = image_aux_list

        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args,  # pyre-fixme
) -> Dict:  # pyre-fixme
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = LazySupervisedDataset(
    #     tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    # )
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_paths=data_args.train_paths, data_args=data_args
    )
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_paths=data_args.eval_paths, data_args=data_args
    )
    
    data_collator_kwargs = {
        "tokenizer": tokenizer,
    }

    if hasattr(data_args, "image_token_len"):
        data_collator_kwargs["image_token_len"] = data_args.image_token_len

    if hasattr(data_args, "vision_tower_aux_token_len_list"):
        data_collator_kwargs["image_aux_token_len_list"] = (
            data_args.vision_tower_aux_token_len_list
        )
    else:
        data_collator_kwargs["image_aux_token_len_list"] = [data_args.image_token_len]

    if hasattr(data_args, "image_position"):
        data_collator_kwargs["image_position"] = data_args.image_position

    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)  # pyre-fixme

    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )