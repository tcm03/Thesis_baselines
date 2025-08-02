import evaluate
import os
import json
import numpy as np

INPUT_PATH = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_1/final_txtcls_txteval_eval_log-epoch1-step1520.json"
OUTPUT_PATH = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_1/auto_caption_evaluation-epoch1-step1520.json"

def main():
    predictions = []
    references = []
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
        for item in data:
            predictions.append(item["gen_pred"])
            references.append(item["reference"])

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    # 1. BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    # 2. ROUGE score
    rouge_score = rouge.compute(predictions=predictions, references=references)
    # 3. Meteor score
    meteor_score = meteor.compute(predictions=predictions, references=references)
    # # 4. BERTScore
    bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")
    # aggregate mean precision, recall and f1 of bertscore
    bertscore_score["precision"] = float(np.mean(bertscore_score["precision"]))
    bertscore_score["recall"] = float(np.mean(bertscore_score["recall"]))
    bertscore_score["f1"] = float(np.mean(bertscore_score["f1"]))

    results = {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "meteor": meteor_score,
        "bertscore": bertscore_score
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()