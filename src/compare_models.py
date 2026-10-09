
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
df = pd.read_csv("results/model_comparison.csv")

# Create folder for graphs
output_dir = Path("results/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Accuracy comparison
plt.figure(figsize=(9, 5))
plt.bar(df["Algorithm"], df["Accuracy"] * 100)
plt.title("Accuracy Comparison of ML Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
plt.show()

# 2. F1-score comparison
plt.figure(figsize=(9, 5))
plt.bar(df["Algorithm"], df["F1-score"] * 100)
plt.title("F1-score Comparison of ML Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Macro F1-score (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(output_dir / "f1_comparison.png", dpi=300)
plt.show()

# 3. Training time comparison
plt.figure(figsize=(9, 5))
plt.bar(df["Algorithm"], df["Training Time (seconds)"])
plt.title("Training Time Comparison of ML Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(output_dir / "training_time_comparison.png", dpi=300)
plt.show()

print("\nGraphs saved in:", output_dir)