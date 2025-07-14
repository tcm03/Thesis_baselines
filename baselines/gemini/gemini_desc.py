from google import genai
import os
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
from tqdm import tqdm
import time
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

class FewshotSample:

    def __init__(self, description: str, engagement_label: str):
        self.description = description
        self.engagement_label = engagement_label

    def to_str(self):
        return f"Description: {self.description}\nEngagement Label: {self.engagement_label}"

fewshot_samples = [
    FewshotSample(
        # video_path: train/train_21/dacc53c0b396a913987128a11958e5f9.mp4
        description="The video begins with a close-up shot of a pair of tan and white sneakers on a gray carpet. Text overlayed on the video reads, \"I fell asleep in my bathtub and flooded my entire apartment \ud83e\udd74.\" The camera then pans up to show the carpet, which is wet and waterlogged. The camera continues to pan across the room, showing the extent of the flooding. The sound of a water pump can be heard. The video ends with a shot of the flooded room. The word \"Oops\" is written on the bottom of the screen.",
        engagement_label="engaged"
    ),
    FewshotSample(
        # video_path: train/train_7/8787be2d403e980fd2737ac42e6f25b7.mp4
        description="The video appears to be a Snapchat video taken from inside a vehicle. It captures a scene in Lewisburg, Tennessee, where two birds are engaging in a fight on a paved road. The video's text overlay humorously claims that the birds are CIA-trained and are attacking each other instead of spying. The video ends with the text \"Birds are not real\" and the location \"Lewisburg, Tennessee\". The background includes houses, trees, and grass, suggesting a residential area.",
        engagement_label="not engaged"
    ),
    FewshotSample(
        # video_path: train/train_16/4aac2790511b3fa304e71f3e24876080.mp4
        description="A young woman is walking down a wet sidewalk on a rainy day. She is wearing a bright pink coat and a pink hat, and she is holding a pink umbrella with white polka dots. She is also carrying a small orange purse. She is smiling and looking at the camera. The sidewalk is wet and there are puddles of water. There are buildings and shops in the background. The camera follows the woman as she walks down the sidewalk.",
        engagement_label="neutral"
    )
]

engagement_mappings = {
    "not engaged": 0,
    "neutral": 1,
    "engaged": 2
}

def exponential_backoff_retry(func, max_retries=5, base_delay=1, max_delay=60):
    """
    Retries a function with exponential backoff and jitter.

    Parameters:
    - func: The function to retry.
    - max_retries: Maximum number of retries.
    - base_delay: Initial delay between retries in seconds.
    - max_delay: Maximum delay between retries in seconds.

    Returns:
    - The result of the function if successful.

    Raises:
    - The last exception raised by the function if all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                logging.error(f"All {max_retries} attempts failed.")
                raise
            else:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = random.uniform(0, delay)
                total_delay = delay + jitter
                logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {total_delay:.2f} seconds.")
                time.sleep(total_delay)

def main(args):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    fewshot_prompt = "\n\n".join([sample.to_str() for sample in fewshot_samples])
    with open(args.json_path, "r") as f:
        data = json.load(f)
    preds = []
    gts = []
    cnt_valid = 0
    for i, item in enumerate(tqdm(data)):
        desc = item["conversations"][1]["value"]
        full_prompt = f"{fewshot_prompt}\n\nDescription: {desc}\nEngagement Label: "

        def call_gemini():
            response = client.models.generate_content(
                model=args.model,
                contents=[full_prompt],
            )
            return response

        try:
            response = exponential_backoff_retry(call_gemini)
            pred = response.text.strip().lower()
            pred_label = int(engagement_mappings[pred])
            preds.append(pred_label)
            gts.append(int(item["label"]))
        except Exception as e:
            logging.error(f"Failed to get response for item: {e}")
            continue
        cnt_valid += 1
        if (i+1) % args.logging_steps == 0:
            logging.info(f"Video {item['video']}, label {item['label']}, pred {pred_label}")

    acc = accuracy_score(gts, preds)
    f1_macro = f1_score(gts, preds, average="macro")
    precision_macro = precision_score(gts, preds, average="macro")
    recall_macro = recall_score(gts, preds, average="macro")
    f1_micro = f1_score(gts, preds, average="micro")
    precision_micro = precision_score(gts, preds, average="micro")
    recall_micro = recall_score(gts, preds, average="micro")
    f1_weighted = f1_score(gts, preds, average="weighted")
    precision_weighted = precision_score(gts, preds, average="weighted")
    recall_weighted = recall_score(gts, preds, average="weighted")
    logging.info(f"Total valid samples: {cnt_valid}/{len(data)}")
    logging.info(f"Accuracy: {acc:.10f}")
    logging.info(f"weighted precision: {precision_weighted:.10f}, recall: {recall_weighted:.10f}, f1: {f1_weighted:.10f}")
    logging.info(f"micro precision: {precision_micro:.10f}, recall: {recall_micro:.10f}, f1: {f1_micro:.10f}")
    logging.info(f"macro precision: {precision_macro:.10f}, recall: {recall_macro:.10f}, f1: {f1_macro:.10f}")

    if args.output_dir is not None:
        results_dir = {
            "model": args.model,
            "num_samples": len(data),
            "num_valid_samples": cnt_valid,
            "accuracy": acc,
            "weighted precision": precision_weighted,
            "weighted recall": recall_weighted,
            "weighted f1": f1_weighted,
            "micro precision": precision_micro,
            "micro recall": recall_micro,
            "micro f1": f1_micro,
            "macro precision": precision_macro,
            "macro recall": recall_macro,
            "macro f1": f1_macro,
        }

        with open(args.output_dir, "w") as f:
            json.dump(results_dir, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()
    main(args)