from clair import clair_gemini
import logging
import os
from google.genai import Client
import json
import time
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_PATH = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_1/final_txtcls_txteval_test_log-testfinal.json"
OUTPUT_PATH = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_1/clair_test_log-testfinal.json"


def evaluate_sample(client, sample) -> dict:
    video_path = sample["video_path"]
    gen_caption = sample["gen_pred"]
    reference = sample["reference"]
    score, reason = clair_gemini(
        client=client,
        candidates=[gen_caption],
        targets=[reference],
        # model_name="gemini-2.5-flash-lite-preview-06-17",
        model_name="gemini-2.0-flash-lite", # more lightweight for unimportant runs to save cost
        max_retries=5,
        backoff_factor=1.0,
        max_tokens=256,
    )
    return {
        "video_path": video_path,
        "gen_caption": gen_caption,
        "reference": reference,
        "score": score,
        "reason": reason
    }

def main():
    logging.basicConfig(level=logging.INFO)
    client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

    with open(DATA_PATH) as f:
        data = json.load(f)

    cpu = os.cpu_count() or 1
    max_workers = min(32, cpu * 2 + 1, len(data))

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_sample, client, sample): sample
            for sample in data
        }
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                res = future.result()
                results.append(res)
                logging.info(f"Done: {res['video_path']} → score={res['score']:.3f}")
            except Exception as e:
                logging.error(f"Error on {sample['video_path']}: {e}")
                results.append({
                    "video_path": sample.get("video_path"),
                    "gen_caption": sample.get("gen_pred"),
                    "reference": sample.get("reference"),
                    "score": None,
                    "reason": f"ERROR: {e}"
                })

    duration = time.time() - start
    logging.info(f"Completed {len(data)} samples in {duration:.1f}s")

    avg_score = sum(r["score"] for r in results if r["score"] is not None) / \
                max(1, sum(1 for r in results if r["score"] is not None))

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"avg_score": avg_score, "results": results}, f, indent=4)

    logging.info(f"Average score: {avg_score:.4f}")

if __name__ == "__main__":
    main()