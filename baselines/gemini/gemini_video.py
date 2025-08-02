from google import genai
from google.genai import types, errors
import os
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import logging
from tqdm import tqdm
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

full_prompt = """
You are given a user‑generated social media video. Your task is twofold:

1. Implicitly consider the video across the following nine dimensions:
    a. **Content** – “Please implicitly analyze how the video's opening moments and storytelling elements contribute to this level of engagement. Examine aspects such as the initial hook, pacing, and narrative structure that may influence viewer attention.”
    b. **Emotion** – “Please examine the emotional tones conveyed throughout the video. Consider how these emotions might impact viewer engagement, and whether they align with the video's overall message.”
    c. **Visual** – “Please evaluate the visual elements, including color schemes, camera angles, and special effects. Think about how these stylistic choices may affect the video's appeal and viewer retention.”
    d. **Audio** – “Please assess the audio components, such as background music, sound effects, and vocal delivery. Analyze how these auditory elements contribute to the video's engagement level.”
    e. **Trend** – “Please determine if the content aligns with current social media trends or cultural moments. Consider how this alignment, or lack thereof, might influence viewer engagement.”
    f. **Interactivity** – “Please explore the interactive elements present, such as calls to action, challenges, or prompts for viewer participation. Evaluate how these features may affect engagement.”
    g. **Authenticity** – “Please analyze the authenticity and relatability of the content. Consider factors like the creator's presentation style, language, and subject matter in relation to viewer connection.”
    h. **Information** – “Please assess the informational or practical value offered. Consider how the content provides utility or knowledge to the viewer and its potential impact on engagement.”
    i. **Quality** – “Please evaluate the technical aspects of production, including video resolution, editing quality, and overall polish. Analyze how these factors may influence viewer engagement.”

2. After analyzing each aspect, combine your insights to **classify the overall engagement level** of the video. Output rationale and finally output one of the three labels:

    - **not engaged**
    - **neutral**
    - **engaged**

    Please stop after outputting one of the three labels above.
"""

FewshotVideos = [
    tuple([
        "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_16/4aac2790511b3fa304e71f3e24876080.mp4", # video_path
        full_prompt, # prompt
        """
Here's an analysis of the video across the nine dimensions, followed by an overall engagement classification:

*   **content:** the video features a person walking in the rain, using a pink umbrella and wearing a pink coat and hat. the content is straightforward and doesn't have a strong hook or narra
tive. the pacing is slow, and the narrative is minimal. this might not immediately grab a viewer's attention.
*   **emotion:** the video evokes a sense of normalcy and perhaps a touch of cheerfulness due to the bright colors used. there's a lack of strong emotional peaks or valleys. the emotion is ne
utral.
*   **visual:** the video uses bright, contrasting colors (pink, blue, orange) against a gray, rainy backdrop. the camera work is simple, with a static shot. the visual appeal is moderately e
ngaging due to the color contrast.
*   **audio:** the audio likely includes the sound of rain and possibly some ambient street noise. there is no background music or voiceover. the audio is not particularly engaging.
*   **trend:** the video doesn't obviously align with any current social media trends.
*   **interactivity:** there are no interactive elements.
*   **authenticity:** the video feels authentic; it's a simple depiction of someone walking in the rain. the relatability depends on the viewer's personal experiences.
*   **information:** the video doesn't provide any informational or practical value.
*   **quality:** the video quality appears to be adequate, but nothing special.

**overall engagement level:** neutral
        """
    ]),
    tuple([
        "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_21/dacc53c0b396a913987128a11958e5f9.mp4",
        full_prompt,
        """
Here's an analysis of the social media video across the provided dimensions:
                                                                                                                                                                                               ht feel empathy or amusement.
**a. content:** the video starts with a relatable "oops" moment – the aftermath of a flooded apartment. the narrative is simple: a person fell asleep in the bathtub and caused a flood. the pa
cing is quick, immediately showing the damage. this directness and relatable scenario likely grab viewers' attention.                                                                          partment creates a visual impa
                                                       
**b. emotion:** the primary emotion is likely a mix of humor and slight distress. the "oops" text and the use of an emoji suggest a lighthearted approach to a stressful situation. viewers mig
ht feel empathy or amusement.

**c. visual:** the video uses a top-down perspective, showing the shoes of the person and then panning out to show the flooded carpet. the visual is simple, but the wide shot of the flooded a
partment creates a visual impact.

**d. audio:** the primary audio element is the sound of a vacuum cleaner, which is likely the creator's attempt to clean up the mess. this adds to the humor and the sense of urgency or effort
.

**e. trend:** the video taps into the trend of sharing relatable, everyday mishaps. the "oops" moment and the quick documentation of the aftermath are common on social media.

**f. interactivity:** the video doesn't have direct interactive elements, but the situation itself invites comments and reactions. viewers might share their own similar experiences.

**g. authenticity:** the video appears authentic because it's a genuine situation. the creator is sharing a real-life problem, which makes it relatable.

**h. information:** the video doesn't offer explicit information but implicitly provides a warning about the dangers of falling asleep in the bathtub.

**i. quality:** the video quality is acceptable, the editing is simple, and the overall polish is sufficient to convey the message.

The relatable content, the humor, the quick pacing, and the visual impact of the flooded apartment all contribute to its potential for high engagement.

**overall engagement level:** the video is likely to be **engaged**.
        """
    ]),
    tuple([
        "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_8/b7d5ba11a489790e59293e50dcbaa9e6.mp4",
        full_prompt,
        """
Here's an analysis of the video across the nine dimensions, followed by an overall engagement classification:                                                                                                           
                                                                                                                                                                                               
**1. dimension analysis:**                                                                                                                                                                     cing is quick, immediately sho
                                                                                                                                                                                               
*   **content:** the video shows a woman getting her makeup done. the initial hook is the visual of the makeup artist applying product, but the pacing is slow. the narrative structure is simp
le, lacking a clear story arc.                                                                                                                                                                 ht feel empathy or amusement.
*   **emotion:** the video conveys a relaxed and possibly slightly stressed emotion. the woman is getting ready, which can be a stressful experience. this emotional tone might not be universa
lly engaging.                                                                                                                                                                                  partment creates a visual impa
*   **visual:** the video has a clean, well-lit aesthetic. the camera angle is static. the visual elements are professional but not particularly dynamic or attention-grabbing.
*   **audio:** the audio includes the sound of the makeup artist, and the woman talking. the audio is clear, but there is no background music or sound effects to enhance engagement.          
*   **trend:** the content is not aligned with any obvious social media trends.
*   **interactivity:** there are no interactive elements like calls to action or prompts for viewer participation.
*   **authenticity:** the video appears authentic in that it shows a real-life scenario. the woman's presentation style is relatable, but the subject matter might not resonate with a broad au
dience.                                        
*   **information:** the video offers no informational or practical value beyond a glimpse into a makeup routine.                                                                    
*   **quality:** the video has good technical quality, with clear visuals and audio. the editing is minimal.                                                                                   
                                                                                                                                                                                               
**2. overall engagement classification:**
                                                                                                                                                                                               
the video is **not engaged**. 
        """
    ])
]
    
def extract_label(text: str) -> str:
    # Find _all_ occurrences
    labels = re.findall(r'\b(not engaged|neutral|engaged)\b', text, flags=re.IGNORECASE)
    if not labels:
        raise ValueError(f"No valid label found in '{text}'")
    # Take the last one and normalize
    return labels[-1].lower()

def request_gemini(
    client,
    target_video_path: str,
    target_prompt: str,
    model_name: str,
    example_videos: List[Tuple[str, str, str]] = None,  # [(video_path, prompt, response), …]
    max_retries: int = 5,
    backoff_factor: float = 1.0,
):
    """
    Sends a few-shot prompt with 3 examples, each containing:
       - video, prompt, and example response (including rationale + label),
    followed by the target video & prompt.
    """
    parts: List[types.Part] = []
    if example_videos is not None:
        # few-shot prompting
        for video_path, ex_prompt, ex_response in example_videos:
            assert os.path.exists(video_path)
            video_bytes = open(video_path, 'rb').read()
            parts.append(types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')))
            parts.append(types.Part(text=f"<EXAMPLE>\nPrompt: {ex_prompt}\nResponse: {ex_response}\n</EXAMPLE>"))

    assert os.path.exists(target_video_path)
    video_bytes = open(target_video_path, 'rb').read()
    parts.append(types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')))
    parts.append(types.Part(text=f"<TARGET>\nPrompt: {target_prompt}\nResponse:"))

    # Send multimodal request
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=1024,
                    top_k=5,
                )
            )
            return response.text
        except errors.ServerError as e:
            if attempt == max_retries:
                logging.error("Max retries reached.")
                raise
            wait_time = backoff_factor * (2 ** (attempt - 1))
            logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

def process_item(item, args, client, full_prompt, examples, engagement_mappings):
    """Encapsulate the per‑video logic so it can be called in a thread."""
    video_path = item["video"]
    full_path = os.path.join(args.base_path, video_path)
    if not os.path.exists(full_path):
        logging.warning(f"Video file {full_path} does not exist. Skipping.")
        return None  # signal skip

    try:
        raw = request_gemini(
            client,
            target_video_path=full_path,
            target_prompt=full_prompt,
            model_name=args.model,
            example_videos=examples,
        ).lower().strip()
        logging.info(f"[{video_path}] Response: {raw}")
        label_str = extract_label(raw)
    except Exception as e:
        logging.error(f"[{video_path}] failed: {e}")
        return None

    pred = engagement_mappings[label_str]
    gt   = int(item["label"])
    return (video_path, gt, pred)

def main(args):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    with open(args.json_path, "r") as f:
        data = json.load(f)
    # # Below are the 3 picked examples from the snapugc_0 training set for 3-shot prompting
    # data = [
    #     {
    #         "video": "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_21/dacc53c0b396a913987128a11958e5f9.mp4",
    #         "label": "2",
    #         "conversations": [
    #             {
    #                 "from": "human",
    #                 "value": "<image>\nDescribe this video in detail with predicted engagement level to viewers.\n"
    #             },
    #             {
    #                 "from": "gpt",
    #                 "value": "The video kicks off with a close-up of stylish beige sneakers, complete with white soles, resting on a textured gray surface that looks like cozy carpet or a soft m"
    #             }
    #         ]
    #     },
    #     {
    #         "video": "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_16/4aac2790511b3fa304e71f3e24876080.mp4",
    #         "label": "1",
    #         "conversations": [
    #             {
    #                 "from": "human",
    #                 "value": "<image>\nDescribe this video in detail with predicted engagement level to viewers.\n"
    #             },
    #             {
    #                 "from": "gpt",
    #                 "value": "The video features a series of moments captured on a rainy day within an urban landscape. It begins with a person standing under a charming pink umbrella on a wet s>"
    #             }
    #         ]
    #     },
    #     {
    #         "video": "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_8/b7d5ba11a489790e59293e50dcbaa9e6.mp4",
    #         "label": "0",
    #         "conversations": [
    #             {
    #                 "from": "human",
    #                 "value": "<image>\nDescribe this video in detail with predicted engagement level to viewers.\n"
    #             },
    #             {
    #                 "from": "gpt",
    #                 "value": "The video presents a sequence of moments in a modern room, featuring a person in a white robe who appears to be going through a routine. It starts with the individu>"
    #             }
    #         ]
    #     }
    # ]
    preds = []
    gts = []
    cnt_valid = 0
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    engagement_mappings = {
        "not engaged": 0,
        "neutral": 1,
        "engaged": 2
    }
    reverse_engagement_mappings = {
        0: "not engaged",
        1: "neutral",
        2: "engaged"
    }
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Submit all tasks
        futures = {
            # ex.submit(process_item, item, args, client, full_prompt, FewshotVideos, engagement_mappings): item
            ex.submit(process_item, item, args, client, full_prompt, None, engagement_mappings): item
            for item in data
        }

        # As each finishes, collect results
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing videos")):
            item = futures[fut]
            res = fut.result()
            if res is None:
                continue
            video_path, gt, pred = res
            cnt_valid += 1
            preds.append(pred)
            gts.append(gt)

            if cnt_valid % args.logging_steps == 0:
                logging.info(f"[{video_path}] GT={gt}, PRED={pred}")

    target_names = ['not engaged', 'neutral', 'engaged']
    report = classification_report(gts, preds, target_names=target_names, output_dict=True)
    acc = report["accuracy"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    f1_macro        = macro["f1-score"]
    precision_macro = macro["precision"]
    recall_macro    = macro["recall"]
    # f1_micro        = f1_score(gts, preds, average="micro")
    # precision_micro = precision_score(gts, preds, average="micro")
    # recall_micro    = recall_score(gts, preds, average="micro")
    f1_weighted     = weighted["f1-score"]
    precision_w     = weighted["precision"]
    recall_w        = weighted["recall"]

    # Logging results
    
    logging.info(f"Valid responses: {cnt_valid}/{len(data)}")
    logging.info(f"Classification report:\n{classification_report(gts, preds, target_names=target_names)}")

    if args.output_dir is not None:
        results_dir = {
            "description:": "Gemini video engagement classification results on the testing set with one-shot prompting video.",
            "model": args.model,
            "json_path": args.json_path,
            "full_prompt": full_prompt,
            "number of samples": len(data),
            "number of valid samples": cnt_valid,
            "accuracy": acc,
            "weighted precision": precision_w,
            "weighted recall": recall_w,
            "weighted f1": f1_weighted,
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
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()
    main(args)