import os
import json

if __name__ == "__main__":
    src_path = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_engcaption.json"
    dst_path = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_engcaption_cls.json"
    with open(src_path, "r") as f:
        data = json.load(f)
    for item in data:
        assert "conversations" in item, "conversations not found in item"
        for conv in item["conversations"]:
            assert "from" in conv and "value" in conv, "from or value not found in conv"
            if conv["from"] == "human":
                prompt = conv["value"]
                if not prompt.endswith("<cls>"):
                    prompt += "<cls>"
                conv["value"] = prompt
    with open(dst_path, "w") as f:
        json.dump(data, f, indent=4)
