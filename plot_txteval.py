import json
import os
import matplotlib.pyplot as plt

# Global variables for input and output paths
INPUT_JSON_PATH = 'checkpoints/longvu_llama_snapugc0_txtcls_analysis/train_perf_txtcls_txteval.json'
OUTPUT_PLOTS_DIR = 'images/txtcls_txteval'

def load_metrics(json_path):
    """
    Load metrics from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_metric(data, metric_key, sub_keys=None):
    """
    Extract metric values over epochs.
    If sub_keys is provided, extract each sub-metric.
    """
    epochs = []
    metrics = {}

    for entry in data:
        epoch = entry.get('epoch')
        if epoch is None:
            continue
        epochs.append(epoch)

        metric_data = entry.get(metric_key, {})
        if sub_keys:
            for sub_key in sub_keys:
                if sub_key not in metrics:
                    metrics[sub_key] = []
                metrics[sub_key].append(metric_data.get(sub_key, None))
        else:
            if metric_key not in metrics:
                metrics[metric_key] = []
            metrics[metric_key].append(metric_data if isinstance(metric_data, (int, float)) else metric_data.get(metric_key, None))

    return epochs, metrics

def plot_metrics(epochs, metrics, metric_name, output_dir):
    """
    Plot metrics over epochs and save the plot.
    """
    plt.figure(figsize=(10, 6))
    for key, values in metrics.items():
        plt.plot(epochs, values, marker='o', label=key.upper())
    plt.title(f'{metric_name.upper()} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plot_filename = f"{metric_name.lower()}_vs_epoch.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

def main():
    # Load data
    data = load_metrics(INPUT_JSON_PATH)

    # Define metrics and their sub-keys
    metrics_info = {
        'bleu': ['bleu'],
        'rouge': ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        'meteor': ['meteor'],
        'bertscore': ['precision', 'recall', 'f1']
    }

    for metric, sub_keys in metrics_info.items():
        epochs, metric_values = extract_metric(data, metric, sub_keys)
        if epochs and metric_values:
            plot_metrics(epochs, metric_values, metric, OUTPUT_PLOTS_DIR)
            print(f"Plot saved for {metric.upper()} at {OUTPUT_PLOTS_DIR}")
        else:
            print(f"Warning: No data found for metric '{metric}'.")

if __name__ == "__main__":
    main()
