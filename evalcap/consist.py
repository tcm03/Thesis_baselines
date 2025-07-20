import os
import json
import logging
from google import genai
from google.genai import Client, types, errors
import time
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging with line numbers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

INPUT_PATH = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_0/final_txtcls_txteval_eval_log-epoch0-step760.json"
OUTPUT_PATH = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_0/alignment_analysis.json"

ALIGNMENT_PROMPT = """\
You are an expert assistant whose mission is to **detect engagement signals** in a video description.

Definitions:
- **engaged**: The text explicitly mentions that the video is engaged.
- **neutral**: The text does not mention engagement, or explicitly mentions that the video is neutral or has a neutral tone/ atmosphere.
- **not engaged**: The text explicitly mentions that the video is not engaged.

Instructions:
1. Analyze the description.
2. Output exactly one of the following labels based strictly on definitions above:
engaged
neutral
not engaged

Please don't output anything else.

### Few‑shot examples

Description: {engaged_description}
Output: engaged

Description: {neutral_description}
Output: neutral

Description: {not_engaged_description}
Output: not engaged

### Now classify:
Description: {description}
Output:
"""

engagement2int = {
    "not engaged": 0,
    "neutral": 1,
    "engaged": 2,
}

def filter_response(s: str) -> str:
    # keep only alphabetic characters and spaces
    return ''.join(filter(lambda c: c.isalpha() or c.isspace(), s))

def request_gemini(
    client,
    prompt: str,
    model_name: str = "gemini-2.0-flash-lite",
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    max_tokens: int = 256,
):
    parts = [types.Part(text=prompt)]

    # Send multimodal request
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                )
            )
            if response is None or response.text is None:
                logging.error("Received empty response from Gemini.")
                continue
            return response.text
        except errors.ServerError as e:
            if attempt == max_retries:
                logging.error("Max retries reached.")
                raise
            wait_time = backoff_factor * (2 ** (attempt - 1))
            logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise

def main():
    logging.info(f"READING INPUT DATA ...")
    with open(INPUT_PATH, 'r') as f:
        eval_log = json.load(f)
    logging.info(f"DONE READING INPUT DATA")

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    cpu = os.cpu_count() or 1
    max_workers = min(32, cpu * 2 + 1, len(eval_log))
    cnt_align = 0
    cnt_valid = 0
    outputs = {
        "alignment_score": None,
        "details": []
    }

    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {}
        for sample in eval_log:
            alignment_prompt = ALIGNMENT_PROMPT.format(
                # no_engagement_description="The video shows a shirtless man in a park performing a series of pull-ups on a set of parallel bars. He is wearing white pants and black shoes, and his physique is well-defined, with a visible six-pack abs. The park is green and has trees, and there are other people in the background, some of whom are also engaged in physical activities. The man maintains a consistent form throughout the video, with his arms extended and legs straight, and his body parallel to the bars. The camera angle is fixed, providing a clear view of the man's upper body and the pull-up bars.",
                engaged_description="The video kicks off with a close-up of a character in a white lab coat, sporting a stethoscope around their neck, and a mustache, set against a plain wall. The character's expression shifts from neutral to one of surprise, and then to a more contemplative look, as the text overlays change to reflect their thoughts. The scene then transitions to a hospital room, where the same character is now standing beside a bed, with a new character in a green shirt and a bandaged arm, hinting at a narrative connection. The hospital room is filled with a sense of urgency, as the text overlays continue to evolve, keeping viewers engaged in the unfolding story. The video seamlessly moves from a close-up of a character's reaction to a broader hospital setting, maintaining a consistent tone and style throughout.",
                neutral_description="The video presents a tranquil outdoor scene featuring a grassy area with a concrete curb and a snake resting on the ground. Initially, the snake is seen coiled on the grass, with its body blending into the surrounding environment. As the video continues, the snake slowly begins to move, its body extending along the concrete curb, showcasing its sleek, dark scales. The camera angle shifts slightly to capture the snake's movement, maintaining a steady focus on its journey along the curb. The overall atmosphere remains calm and serene, with no significant changes in the background or lighting, resulting in a neutral engagement.",
                not_engaged_description="The video presents a series of static shots featuring a horse and rider on a dirt path, likely in a rural or park setting. The horse, with its dark mane and tail, is captured from behind, showcasing its muscular build and the rider's attire, which includes a helmet and a jacket. The background is consistent, with lush greenery and a fence lining the path, creating a serene atmosphere. However, the camera remains fixed in place, offering little variation in perspective or action. The rider's posture and the horse's movements are minimal, contributing to a sense of stillness and lack of dynamic engagement. Thus, the video is not engaged.",
                description=sample["gen_pred"]
            )
            sub_var = executor.submit(request_gemini, client, alignment_prompt)
            future_to_sample[sub_var] = sample

        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                response = future.result()
                if response is not None:
                    cnt_valid += 1
                    response = response.strip().lower()
                    response = filter_response(response)
                video_path = sample["video_path"]
                outputs["details"].append({
                    "video_path": video_path,
                    "response": response,
                })
                gold_label = int(sample["gold_label"])
                pred_label = None
                if response is not None and response in engagement2int:
                    pred_label = engagement2int[response]
                if pred_label is not None and gold_label == pred_label:
                    cnt_align += 1
            except Exception as e:
                logging.error(f"Error on {sample['video_path']}: {e}")
                raise
                
    duration = time.time() - start
    logging.info(f"Completed {len(eval_log)} samples in {duration:.1f}s")

    alignment_score = cnt_align / len(eval_log)
    logging.info(f"Alignment percentage: {alignment_score*100:.4f}%")
    outputs["alignment_score"] = alignment_score
    outputs["valid"] = f"{cnt_valid}/{len(eval_log)}"

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    main()