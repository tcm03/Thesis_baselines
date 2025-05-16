from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class CustomCausalLMOutputWithPast(CausalLMOutputWithPast):
    labels: Optional[torch.LongTensor] = None