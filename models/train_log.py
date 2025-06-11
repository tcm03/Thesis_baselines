from typing import Dict, Any

class TrainProgressLog:
    def __init__(
        self,
        run_type: str,
        epoch: float,
        step: int, 
        loss: float, 
        grad_norm: float, 
        learning_rate: float,
        video_path: str = None,
        cls_pred: str = None,
        gen_pred: str = None
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

class PerfMetrics:
    def __init__(
        self, 
        epoch: float,
        step: int,
        accuracy: float, 
        precision: Dict[str, float], 
        recall: Dict[str, float], 
        f1: Dict[str, float],
        loss: float = None,
        bleu: Dict[str, Any] = None,
        rouge: Dict[str, Any] = None,
        meteor: Dict[str, Any] = None,
        bertscore: Dict[str, Any] = None
    ):
        self.epoch = epoch
        self.step = step
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.loss = loss
        self.bleu = bleu
        self.rouge = rouge
        self.meteor = meteor
        self.bertscore = bertscore

    def to_dict(self) -> Dict[str, Any]:
        return_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "bleu": self.bleu,
            "rouge": self.rouge,
            "meteor": self.meteor,
            "bertscore": self.bertscore
        }
        if self.loss is not None:
            return_dict["loss"] = self.loss
        return return_dict

class EvalProgressLog:

    def __init__(
        self,
        epoch: float,
        step: int,
        video_path: str,
        cls_pred: str,
        gen_pred: str
    ):
        self.epoch = epoch
        self.step = step
        self.video_path = video_path
        self.cls_pred = cls_pred
        self.gen_pred = gen_pred

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "video_path": self.video_path,
            "cls_pred": self.cls_pred,
            "gen_pred": self.gen_pred
        }