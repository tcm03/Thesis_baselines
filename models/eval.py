import torch
import torch.distributed as dist
import os
from typing import List
from sklearn.metrics import classification_report
from models.train_log import PerfMetrics
import logging
import numpy as np

def evaluate_perf(
    device_preds: List[torch.Tensor],
    device_gold_labels: List[torch.Tensor],
    device_loss: float = None,
    device_samples: int = None,
    predictions: List[str] = None,
    references: List[str] = None,
    prefix: str = "Train",
    **kwargs
):
    is_distributed = int(os.environ.get("RANK", -1)) != -1
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0 # main process for logging, checkpointing, etc.

    agg_loss = None
    if device_loss is not None and device_samples is not None:
        total_loss_tensor = torch.tensor(device_loss, device=device)
        total_samples_tensor = torch.tensor(device_samples, device=device)
        if is_distributed:
            # aggregate losses across devices
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        agg_loss = total_loss_tensor.item() / total_samples_tensor.item()

    preds = torch.cat(device_preds, dim=0)
    gold_labels = torch.cat(device_gold_labels, dim=0)
    # Gather predictions from all processes
    if is_distributed:
        all_preds = [torch.zeros_like(preds) for _ in range(ddp_world_size)]
        all_gold_labels = [torch.zeros_like(gold_labels) for _ in range(ddp_world_size)]
        dist.all_gather(all_preds, preds)
        dist.all_gather(all_gold_labels, gold_labels)
        preds = torch.cat(all_preds, dim=0)
        gold_labels = torch.cat(all_gold_labels, dim=0)
    
    preds_np = preds.cpu().numpy()
    gold_labels_np = gold_labels.cpu().numpy()
    cur_perf = None
    if master_process:
        report = classification_report(gold_labels_np, preds_np, output_dict=True)
        accuracy = report["accuracy"]
        prec_w = report["weighted avg"]["precision"]
        recall_w = report["weighted avg"]["recall"]
        f1_w = report["weighted avg"]["f1-score"]
        prec_macro = report["macro avg"]["precision"]
        recall_macro = report["macro avg"]["recall"]
        f1_macro = report["macro avg"]["f1-score"]

        cur_perf = PerfMetrics(
            epoch=kwargs.get("epoch", None),
            step=kwargs.get("step", None),
            accuracy=accuracy,
            precision={
                "weighted": prec_w,
                "macro": prec_macro
            },
            recall={
                "weighted": recall_w,
                "macro": recall_macro
            },
            f1={
                "weighted": f1_w,
                "macro": f1_macro
            },
            loss=agg_loss
        )

        # 1. BLEU score
        if "bleu" in kwargs and predictions is not None and references is not None:
            bleu = kwargs["bleu"]
            bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
            cur_perf.bleu = bleu_score
        # 2. ROUGE score
        if "rouge" in kwargs and predictions is not None and references is not None:
            rouge = kwargs["rouge"]
            rouge_score = rouge.compute(predictions=predictions, references=references)
            cur_perf.rouge = rouge_score
        # 3. Meteor score
        if "meteor" in kwargs and predictions is not None and references is not None:
            meteor = kwargs["meteor"]
            meteor_score = meteor.compute(predictions=predictions, references=references)
            cur_perf.meteor = meteor_score
        # 4. BERTScore
        if "bertscore" in kwargs and predictions is not None and references is not None:
            bertscore = kwargs["bertscore"]
            bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")
            # aggregate mean precision, recall and f1 of bertscore
            bertscore_score["precision"] = float(np.mean(bertscore_score["precision"]))
            bertscore_score["recall"] = float(np.mean(bertscore_score["recall"]))
            bertscore_score["f1"] = float(np.mean(bertscore_score["f1"]))
            cur_perf.bertscore = bertscore_score

        if agg_loss is not None:
            logging.info(f"{prefix} loss: {agg_loss:.10f}")
        logging.info(f"{prefix} accuracy: {accuracy:.10f}")
        logging.info(f"{prefix} weighted precision: {prec_w:.10f}, recall: {recall_w:.10f}, f1: {f1_w:.10f}")
        logging.info(f"{prefix} macro precision: {prec_macro:.10f}, recall: {recall_macro:.10f}, f1: {f1_macro:.10f}")
        logging.info(f"{prefix} bleu: {bleu_score if 'bleu' in kwargs else 'N/A'}")
        logging.info(f"{prefix} rouge: {rouge_score if 'rouge' in kwargs else 'N/A'}")
        logging.info(f"{prefix} meteor: {meteor_score if 'meteor' in kwargs else 'N/A'}")
        logging.info(f"{prefix} bertscore: {bertscore_score if 'bertscore' in kwargs else 'N/A'}")

    return cur_perf