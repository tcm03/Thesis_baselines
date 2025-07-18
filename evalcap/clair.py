import json
import logging
import re
import os
from typing import List, Optional, Tuple
from google.genai import Client, types, errors

_engine_cache = {}

_CLAIR_PROMPT = """\
You are trying to tell if a candidate set of captions is describing the same video as a reference set of captions.
Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same video as the reference set? (JSON format, with a key "score", value between 0 and 100, and a key "reason" with a string value.)
"""

def clair_gemini(
    client: Client,
    candidates: List[str],
    targets: List[str],
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    max_tokens: int = 256,
) -> Tuple[float, Optional[str]]:
    """Evaluate candidate vs. reference captions using Gemini."""
    candidate_statements = "".join(f"- {c}\n" for c in candidates)
    target_statements = "".join(f"- {t}\n" for t in targets)
    prompt = _CLAIR_PROMPT.format(
        candidate_statements=candidate_statements,
        target_statements=target_statements,
    )
    parts = [types.Part(text=prompt)]
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens
                ),
            )
            text = response.text.strip()
            logging.debug(f"CLAIR–Gemini response: {text}")

            # Try JSON parsing
            try:
                obj = json.loads(text)
                score = float(obj["score"])
                reason = obj.get("reason", None)
            except json.JSONDecodeError:
                # Fallback number parsing
                nums = re.findall(r"\d*\.?\d+", text)
                score = float(nums[0]) if nums else 0.0
                if score < 1.0:
                    score *= 100
                reason_match = re.search(r"(?i)reason[:\s]*(.*)", text)
                reason = reason_match.group(1).strip() if reason_match else None

            return score / 100.0, reason
        except errors.ServerError as e:
            if attempt == max_retries:
                logging.error("Duplicate failure, giving up.")
                raise
            wait = backoff_factor * (2 ** (attempt - 1))
            logging.warning(f"Retry {attempt} after {wait}s: {e}")
            time.sleep(wait)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

    import sys
    with open(sys.argv[1]) as f:
        data = json.load(f)
    for sample in data:
        score, reason = clair_gemini(
            client,
            [sample['test']],
            sample['refs'],
        )
        print(f"Score: {score:.3f}, Reason: {reason}")
