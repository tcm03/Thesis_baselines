from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length
from transformers import Trainer

def get_optimizer(opt_model, training_args):

    # ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
        training_args
    )

    optimizer = optimizer_cls(
        optimizer_grouped_parameters, **optimizer_kwargs
    )

    return optimizer