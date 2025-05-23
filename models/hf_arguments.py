from typing import Optional, List
from dataclasses import dataclass, field
import transformers

@dataclass
class ModelArguments:
    input_model_filename: Optional[str] = field(default=None)
    output_model_filename: Optional[str] = field(default=None)
    checkpoint_fname: Optional[str] = field(default=None)
    
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_lm_head: bool = field(default=False)
    tune_cls_head: bool = field(default=False)
    tune_embed_tokens: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    grid_size: Optional[int] = field(default=8)
    vision_tower_type: Optional[str] = field(default="sam")
    mm_hidden_size: Optional[int] = field(default=256)

    # cambrian
    vision_tower_aux_list: Optional[str] = field(
        default='["siglip/CLIP-ViT-SO400M-14-384", "facebook/dinov2-giant-res378"]'
    )
    vision_tower_aux_token_len_list: Optional[str] = field(default="[576, 576]")
    image_token_len: Optional[int] = field(default=576)
    num_query_group: Optional[int] = field(default=1)
    query_num_list: Optional[str] = field(default="[576]")
    connector_depth: Optional[int] = field(default=3)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    num_of_vision_sampler_layers: Optional[int] = field(default=10)
    start_of_vision_sampler_layers: Optional[int] = field(default=0)
    stride_of_vision_sampler_layers: Optional[int] = field(default=3)

    is_st_sampler: bool = field(default=False)
    highres_connect: bool = field(default=False)
    highres: bool = field(default=False)
    connect_layer: Optional[int] = field(default=2)
    lowres_token: Optional[int] = field(default=8)
    dino_threshold: float = field(default=0.83)
    drop_threshold: float = field(default=0.8)
    frame_pos: bool = field(default=False)
    is_image_newline: bool = field(default=True)


@dataclass
class DataArguments:
    # data_path: Optional[str] = field(default=None)
    train_paths: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Paths to training metadata files"
        }
    )
    eval_paths: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Paths to evaluation metadata files"
        }
    )

    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_position: Optional[int] = field(default=91)
    
    # image_folder: Optional[str] = field(default=None)
    image_folders: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Paths to video datasets"
        }
    )
    
    uniform_sample: bool = field(default=False)
    image_aspect_ratio: str = "square"
    num_points: int = field(default=0)
    video_fps: float = field(default=1)
    use_subtitle: bool = field(default=True)


@dataclass
class CustomTrainingArguments(transformers.TrainingArguments):

    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_text_decoder: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    unfreeze_mm_image_decoder: bool = field(default=False)

    mm_vision_sampler_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    model_max_length: Optional[int] = field(default=8192)

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # If `"epoch"` or `"steps"` is chosen, saving will also be performed at the very end of training, always.
    num_train_epochs: int = field(
        default=1,
        metadata={
            "help": "Number of training epochs"
        }
    ) # HF implements as float (fraction of last epoch), but for the moment let's use int
    # @tcm: I'm worried that some version of transformers still use the deprecated 'evaluation_strategy' arg
    eval_strategy: str = field(
        default="epoch",
        metadata={
            "help": "Evaluation strategy to use during training"
        }
    )
    # save_strategy is inherited from HF
    # warmup_ratio is inherited from HF
    # learning_rate is inherited from HF
    # weight_decay is inherited from HF
    # logging_steps is inherited from HF
    # eval_steps is inherited from HF
    # per_device_train_batch_size is inherited from HF
    # per_device_eval_batch_size is inherited from HF

    # dataloader_num_workers: int = field(
    #     default=0,
    #     metadata={
    #         "help": "Number of workers for the dataloader"
    #     }
    # )

    train_log: Optional[str] = field(default=None)
    train_perf_log: Optional[str] = field(default=None)
    eval_perf_log: Optional[str] = field(default=None)


# @dataclass
# class CustomArguments:
    # input_checkpoint_path: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Path to the checkpoint file to load"
    #     }
    # )

    # output_checkpoint_path: str = field(
    #     metadata={
    #         "help": "Path to save the checkpoint file"
    #     }
    # )

    # data_paths: List[str] = field(
    #     default_factory=list,
    #     metadata={
    #         "help": "Paths to video datasets"
    #     }
    # )

    # train_paths: List[str] = field(
    #     default_factory=list,
    #     metadata={
    #         "help": "Paths to training metadata files"
    #     }
    # )

    # eval_paths: List[str] = field(
    #     default_factory=list,
    #     metadata={
    #         "help": "Paths to evaluation metadata files"
    #     }
    # )

    # @tcm: for saving a checkpoint and loading it the next time
    # model_path: str = field(
    #     metadata={
    #         "help": "Path to the model checkpoint"
    #     }
    # ) 

    # output_file: str = field(
    #     default="entube_tensors.safetensors",
    #     metadata={
    #         "help": "Safetensor file to store embeddings"
    #     }
    # )

    # this arg is for loading CambrianConfig from json file for baselines (baselines.preprocessor.py)
    # config_file: str = field(
    #     default="config.json",
    #     metadata={
    #         "help": "Path to configuration file of encoders parameters"
    #     }
    # )

    # eval_strategy: str = field(
    #     default="epoch",
    #     metadata={
    #         "help": "Evaluation strategy to use during training"
    #     }
    # )

    # save_strategy: str = field(
    #     default="epoch",
    #     metadata={
    #         "help": "Save strategy to use during training"
    #     }
    # )

    # eval_on_final_epoch: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to evaluate on the final epoch"
    #     }
    # )

    # num_train_epochs: int = field(
    #     metadata={
    #         "help": "Number of training epochs"
    #     }
    # )

    # warmup_ratio: float = field(
    #     metadata={
    #         "help": "Warmup ratio for learning rate schedule"
    #     }
    # )

    # learning_rate: float = field(
    #     metadata={
    #         "help": "Learning rate for the optimizer"
    #     }
    # )

    # weight_decay: float = field(
    #     metadata={
    #         "help": "Weight decay for the optimizer"
    #     }
    # )

    # logging_steps: int = field(
    #     metadata={
    #         "help": "Log every X updates steps"
    #     }
    # )

    # eval_steps: int = field(
    #     metadata={
    #         "help": "Evaluate every X updates steps"
    #     }
    # )

    # per_device_train_batch_size: int = field(
    #     default=1,
    #     metadata={
    #         "help": "Per-device train batch size for running"
    #     }
    # )

    # per_device_eval_batch_size: int = field(
    #     default=1,
    #     metadata={
    #         "help": "Per-device eval batch size for running"
    #     }
    # )

    # HF training args has this arg:
    # resume_from_checkpoint (`str`, *optional*):
    # The path to a folder with a valid checkpoint for your model. This argument is not directly used by
    # [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
    # scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
    # resume: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Resume training from the last checkpoint"
    #     }
    # )