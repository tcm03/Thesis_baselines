import time
from huggingface_hub import snapshot_download

for attempt in range(5):
    try:
        snapshot_download(
            repo_id="Vision-CAIR/LongVU_Llama3_2_3B",
            cache_dir="./checkpoints/longvu_llama3_2",
            etag_timeout=60,
        )
        print("✅ Download completed")
        break
    except Exception as e:
        print(f"❌ Attempt {attempt+1} failed: {e}")
        time.sleep(5 * (attempt + 1))
else:
    raise RuntimeError("Failed after 5 download attempts")
