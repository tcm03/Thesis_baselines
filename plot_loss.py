import json
import matplotlib.pyplot as plt

# Load the training log data from the JSON file
with open('checkpoints/longvu_llama_snapugc0_txtcls/train_log_txtcls_1.json', 'r') as f:
    data = json.load(f)

# Extract the steps, loss, gradient norm, and learning rate values
steps = [entry["step"] for entry in data]
losses = [entry["loss"] for entry in data]
grad_norms = [entry["grad_norm"] for entry in data]
learning_rates = [entry["learning_rate"] for entry in data]

# Plot Training Loss vs Step
plt.figure(figsize=(10, 4))
plt.plot(steps, losses, marker='o', linestyle='-')
plt.title("Training Loss vs Step")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_vs_step.png", dpi=300)
plt.close()

# Plot Gradient Norm vs Step
plt.figure(figsize=(10, 4))
plt.plot(steps, grad_norms, marker='o', linestyle='-')
plt.title("Gradient Norm vs Step")
plt.xlabel("Step")
plt.ylabel("Gradient Norm")
plt.grid(True)
plt.tight_layout()
plt.savefig("grad_norm_vs_step.png", dpi=300)
plt.close()

# Plot Learning Rate vs Step
plt.figure(figsize=(10, 4))
plt.plot(steps, learning_rates, marker='o', linestyle='-')
plt.title("Learning Rate vs Step")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_rate_vs_step.png", dpi=300)
plt.close()
