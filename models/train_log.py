from typing import Dict, Any

class TrainProgressLog:
    def __init__(
        self,
        run_type: str,
        epoch: float,
        step: int, 
        loss: float, 
        grad_norm: float, 
        learning_rate: float
    ):
        self.run_type = run_type
        self.epoch = epoch
        self.step = step
        self.loss = loss
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_type": self.run_type,
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate
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
        loss: float = None
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
            "f1": self.f1
        }
        if self.loss is not None:
            return_dict["loss"] = self.loss
        return return_dict
