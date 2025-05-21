import json
import os
import matplotlib.pyplot as plt

# Define the directory to save images
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Load training performance data
with open('checkpoints/longvu_llama_snapugc0_txtcls/train_perf_txtcls_1.json', 'r') as f:
    train_data = json.load(f)

# Load evaluation performance data
with open('checkpoints/longvu_llama_snapugc0_txtcls/eval_perf_txtcls_1.json', 'r') as f:
    eval_data = json.load(f)

# Extract epochs
train_epochs = [entry['epoch'] for entry in train_data]
eval_epochs = [entry['epoch'] for entry in eval_data]

# Extract metrics
train_accuracy = [entry['accuracy'] for entry in train_data]
eval_accuracy = [entry['accuracy'] for entry in eval_data]

train_precision = [entry['precision']['weighted'] for entry in train_data]
eval_precision = [entry['precision']['weighted'] for entry in eval_data]

train_recall = [entry['recall']['weighted'] for entry in train_data]
eval_recall = [entry['recall']['weighted'] for entry in eval_data]

train_f1 = [entry['f1']['weighted'] for entry in train_data]
eval_f1 = [entry['f1']['weighted'] for entry in eval_data]

# Define a function to plot and save metrics
def plot_metric(train_epochs, train_values, eval_epochs, eval_values, metric_name, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_values, marker='o', label='Training')
    plt.plot(eval_epochs, eval_values, marker='s', label='Validation')
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Plot and save each metric
plot_metric(train_epochs, train_accuracy, eval_epochs, eval_accuracy, 'Accuracy', 'accuracy_over_epochs.png')
plot_metric(train_epochs, train_precision, eval_epochs, eval_precision, 'Precision (Weighted)', 'precision_over_epochs.png')
plot_metric(train_epochs, train_recall, eval_epochs, eval_recall, 'Recall (Weighted)', 'recall_over_epochs.png')
plot_metric(train_epochs, train_f1, eval_epochs, eval_f1, 'F1 Score (Weighted)', 'f1_score_over_epochs.png')
