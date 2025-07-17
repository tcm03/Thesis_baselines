from models.supervised_dataset import EvalSupervisedDataset
import transformers
import argparse
import models.conversation_lib as conversation_lib
from models.hf_arguments import ModelArguments, DataArguments, CustomTrainingArguments
import torch
from torch.nn import DataParallel
from models.cambrian_llama import CambrianLlamaForSequenceClassification, CambrianLlamaForCausalLM
import json
from train_ckpt import load_checkpoint

def main(args):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.input_model_filename,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    tokenizer.pad_token_id = 128002
    if args.cls_only:
        model = CambrianLlamaForSequenceClassification.from_pretrained(
            model_args.input_model_filename,
        )
    else:
        model = CambrianLlamaForCausalLM.from_pretrained(
            model_args.input_model_filename,
        )
    model.config.use_cache = False
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        model_args.version
    ]
    if model_args.vision_tower_aux_list is not None:
        # pyre-fixme[16]: `DataClass` has no attribute `unfreeze_mm_vision_tower`.
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_token_len_list`.
        model_args.vision_tower_aux_token_len_list = json.loads(
            model_args.vision_tower_aux_token_len_list
        )
        # pyre-fixme[16]: `DataClass` has no attribute `query_num_list`.
        model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=None,  # FSDP or not, flag should be the same as None to avoid creation error
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()

        if not training_args.unfreeze_mm_vision_tower:
            # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(
                        dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device  # pyre-fixme
                    )
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [  # pyre-fixme
                vision_tower_aux.image_processor
                for vision_tower_aux in vision_tower_aux_list
            ]
        data_args.is_multimodal = True  # pyre-fixme

        model.config.image_aspect_ratio = data_args.image_aspect_ratio  # pyre-fixme
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position  # pyre-fixme
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token  # pyre-fixme

        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.image_token_len = model_args.image_token_len = (  # pyre-fixme
            model_args.image_token_len
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr  # pyre-fixme
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr  # pyre-fixme
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr  # pyre-fixme
        training_args.use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = (
            data_args.vision_tower_aux_token_len_list
        ) = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = model_args.image_token_len
        model.config.is_st_sampler = model_args.is_st_sampler  # pyre-fixme
        data_args.image_token_len = model_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    if training_args.bf16:
        model.to(torch.bfloat16)
    elif training_args.fp16:
        model.to(torch.float16)
    model.to(device)
    # pyre-fixme
    def convert_bn_to_float(model):
        if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
            return model.float()
        for child_name, child in model.named_children():
            model.add_module(child_name, convert_bn_to_float(child))
        return model

    model = convert_bn_to_float(model)
    checkpoint_path = args.model_path
    generator = torch.Generator() # does nothing during testing
    generator.manual_seed(GLOBAL_SEED)
    load_checkpoint(checkpoint_path, training_args, model, generator)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    test_dataset = data_module["test_dataset"]
    data_collator = data_module["data_collator"]

    assert training_args.group_by_modality_length is True, "Group by modality length must be True"
    # Instantiate LengthGroupedSampler
    test_sampler = LengthGroupedSampler(
        batch_size=training_args.per_device_eval_batch_size,
        world_size=ddp_world_size,
        lengths=test_dataset.modality_lengths,
        generator=generator,
        group_by_modality=training_args.group_by_modality_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True, # per-device train batch size = 1 so we won't miss too many samples
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True, # per-device eval batch size = 1 so we won't miss too many samples
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_max_length", type=int, required=True)
    parser.add_argument("--cls_only", action="store_true")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()
    main(args)