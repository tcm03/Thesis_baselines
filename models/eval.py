import torch
import torch.distributed as dist
import os
from typing import List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models.train_log import PerfMetrics
import logging

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
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0 # main process for logging, checkpointing, etc.

    agg_loss = None
    if device_loss is not None and device_samples is not None:
        total_loss_tensor = torch.tensor(device_loss, device=device)
        total_samples_tensor = torch.tensor(device_samples, device=device)
        if ddp:
            # aggregate losses across devices
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        agg_loss = total_loss_tensor.item() / total_samples_tensor.item()

    preds = torch.cat(device_preds, dim=0)
    gold_labels = torch.cat(device_gold_labels, dim=0)
    # Gather predictions from all processes
    if ddp:
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
        accuracy = accuracy_score(gold_labels_np, preds_np)
        prec_w, recall_w, f1_w, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='weighted')
        prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='micro')
        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold_labels_np, preds_np, average='macro')

        cur_perf = PerfMetrics(
            epoch=kwargs.get("epoch", 9999),
            step=kwargs.get("step", 9999),
            accuracy=accuracy,
            precision={
                "weighted": prec_w,
                "micro": prec_micro,
                "macro": prec_macro
            },
            recall={
                "weighted": recall_w,
                "micro": recall_micro,
                "macro": recall_macro
            },
            f1={
                "weighted": f1_w,
                "micro": f1_micro,
                "macro": f1_macro
            },
            loss=agg_loss
        )

        # 1. BLEU score
        if "bleu" in kwargs:
            bleu = kwargs["bleu"]
            bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
            cur_perf.bleu = bleu_score
        # 2. ROUGE score
        if "rouge" in kwargs:
            rouge = kwargs["rouge"]
            rouge_score = rouge.compute(predictions=predictions, references=references)
            cur_perf.rouge = rouge_score
        # 3. Meteor score
        if "meteor" in kwargs:
            meteor = kwargs["meteor"]
            meteor_score = meteor.compute(predictions=predictions, references=references)
            cur_perf.meteor = meteor_score
        # 4. BERTScore
        if "bertscore" in kwargs:
            bertscore = kwargs["bertscore"]
            bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")
            cur_perf.bertscore = bertscore_score

        if agg_loss is not None:
            logging.info(f"{prefix} loss: {agg_loss:.10f}")
        logging.info(f"{prefix} accuracy: {accuracy:.10f}")
        logging.info(f"{prefix} weighted precision: {prec_w:.10f}, recall: {recall_w:.10f}, f1: {f1_w:.10f}")
        logging.info(f"{prefix} micro precision: {prec_micro:.10f}, recall: {recall_micro:.10f}, f1: {f1_micro:.10f}")
        logging.info(f"{prefix} macro precision: {prec_macro:.10f}, recall: {recall_macro:.10f}, f1: {f1_macro:.10f}")
        logging.info(f"{prefix} bleu: {bleu_score if 'bleu' in kwargs else 'N/A'}")
        logging.info(f"{prefix} rouge: {rouge_score if 'rouge' in kwargs else 'N/A'}")
        logging.info(f"{prefix} meteor: {meteor_score if 'meteor' in kwargs else 'N/A'}")
        logging.info(f"{prefix} bertscore: {bertscore_score if 'bertscore' in kwargs else 'N/A'}")

    return cur_perf