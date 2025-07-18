import torch
from typing import Dict, Any, Optional, Union

class TrainProgressLog:
    def __init__(
        self,
        run_type: str,
        loss: float, 
        grad_norm: float, 
        learning_rate: float,
        video_path: str = None,
        cls_pred: str = None,
        gen_pred: str = None,
        epoch: Optional[float] = None,
        step: Optional[int] = None, 
    ):
        self.run_type = run_type
        self.epoch = epoch
        self.step = step
        self.loss = loss
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate

        self.video_path = video_path
        self.cls_pred = cls_pred
        self.gen_pred = gen_pred

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_type": self.run_type,
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "video_path": self.video_path,
            "cls_pred": self.cls_pred,
            "gen_pred": self.gen_pred
        }

class ClsPerfMetrics:
    def __init__(
        self, 
        accuracy: float, 
        precision: Dict[str, float], 
        recall: Dict[str, float], 
        f1: Dict[str, float],
        loss: float = None,
        bleu: Dict[str, Any] = None,
        rouge: Dict[str, Any] = None,
        meteor: Dict[str, Any] = None,
        bertscore: Dict[str, Any] = None,
        epoch: Optional[float] = None,
        step: Optional[int] = None,
    ):
        self.epoch = epoch
        self.step = step
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.loss = loss

    def to_dict(self) -> Dict[str, Any]:
        return_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.loss is not None:
            return_dict["loss"] = self.loss
        return return_dict

class TextPerfMetrics:
    def __init__(
        self, 
        bleu: Dict[str, Any] = None,
        rouge: Dict[str, Any] = None,
        meteor: Dict[str, Any] = None,
        bertscore: Dict[str, Any] = None,
        epoch: Optional[float] = None,
        step: Optional[int] = None,
    ):
        self.epoch = epoch
        self.step = step
        self.bleu = bleu
        self.rouge = rouge
        self.meteor = meteor
        self.bertscore = bertscore

    def to_dict(self) -> Dict[str, Any]:
        return_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "bleu": self.bleu,
            "rouge": self.rouge,
            "meteor": self.meteor,
            "bertscore": self.bertscore
        }
        return return_dict

class EvalProgressLog:

    def __init__(
        self,
        epoch: float,
        step: int,
        video_path: str,
        cls_logits: Union[list, torch.Tensor],
        cls_pred: int,
        gold_label: int,
        gen_pred: str,
        reference: str
    ):
        self.epoch = epoch
        self.step = step
        self.video_path = video_path
        self.cls_logits = cls_logits
        self.cls_pred = cls_pred
        self.gold_label = gold_label
        self.gen_pred = gen_pred
        self.reference = reference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "video_path": self.video_path,
            "cls_logits": self.cls_logits.tolist() if isinstance(self.cls_logits, torch.Tensor) else self.cls_logits,
            "cls_pred": self.cls_pred,
            "gold_label": self.gold_label,
            "gen_pred": self.gen_pred,
            "reference": self.reference
        }