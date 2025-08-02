import json
import os

def main():
    clsonly_path = "/media02/nthuy/Thesis_baselines/checkpoints/final_clsonly_0/final_clsonly_0_test_log-qualitative.json"
    clsonly_result = {
        "correct": [],
        "incorrect": [],
    }
    with open(clsonly_path, 'r') as f:
        clsonly_data = json.load(f)
        for category, items in clsonly_data.items():
            for item in items:
                if int(item["cls_pred"]) == int(item["gold_label"]):
                    clsonly_result["correct"].append(item)
                else:
                    clsonly_result["incorrect"].append(item)
    
    txtcls_path = "/media02/nthuy/Thesis_baselines/checkpoints/final_txtcls_txteval_1/final_txtcls_txteval_test_log-testfinal.json"
    txtcls_result = {
        "correct": [],
        "incorrect": [],
    }
    with open(txtcls_path, 'r') as f:
        txtcls_data = json.load(f)
        for item in txtcls_data:
            if int(item["cls_pred"]) == int(item["gold_label"]):
                txtcls_result["correct"].append(item)
            else:
                txtcls_result["incorrect"].append(item)

    # clsonly wrong, txtcls correct
    comparison_result = {
        "incorrect-incorrect": {
            "count": 0,
            "examples": []
        },
        "incorrect-correct": {
            "count": 0,
            "examples": []
        },
        "correct-correct": {
            "count": 0,
            "examples": []
        },
        "correct-incorrect": {
            "count": 0,
            "examples": []
        }
    }
    for item in clsonly_result["incorrect"]:
        for txt_item in txtcls_result["incorrect"]:
            if item["video_path"] == txt_item["video_path"]:
                comparison_result["incorrect-incorrect"]["count"] += 1
                comparison_result["incorrect-incorrect"]["examples"].append({
                    "video_path": item["video_path"],
                    "gold_label": item["gold_label"],
                    "clsonly_pred": item["cls_pred"],
                    "txtcls_pred": txt_item["cls_pred"],
                })
    for item in clsonly_result["incorrect"]:
        for txt_item in txtcls_result["correct"]:
            if item["video_path"] == txt_item["video_path"]:
                comparison_result["incorrect-correct"]["count"] += 1
                comparison_result["incorrect-correct"]["examples"].append({
                    "video_path": item["video_path"],
                    "gold_label": item["gold_label"],
                    "clsonly_pred": item["cls_pred"],
                    "txtcls_pred": txt_item["cls_pred"],
                })
    for item in clsonly_result["correct"]:
        for txt_item in txtcls_result["correct"]:
            if item["video_path"] == txt_item["video_path"]:
                comparison_result["correct-correct"]["count"] += 1
                comparison_result["correct-correct"]["examples"].append({
                    "video_path": item["video_path"],
                    "gold_label": item["gold_label"],
                    "clsonly_pred": item["cls_pred"],
                    "txtcls_pred": txt_item["cls_pred"],
                })
    for item in clsonly_result["correct"]:
        for txt_item in txtcls_result["incorrect"]:
            if item["video_path"] == txt_item["video_path"]:
                comparison_result["correct-incorrect"]["count"] += 1
                comparison_result["correct-incorrect"]["examples"].append({
                    "video_path": item["video_path"],
                    "gold_label": item["gold_label"],
                    "clsonly_pred": item["cls_pred"],
                    "txtcls_pred": txt_item["cls_pred"],
                })
    # Save the comparison result to a JSON file
    output_path = "comparison_result.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_result, f, indent=4)


if __name__ == "__main__":
    main()