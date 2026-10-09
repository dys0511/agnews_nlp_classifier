
import pandas as pd
from pathlib import Path

results = [
    {
        "Algorithm": "Linear SVM",
        "Accuracy": 0.9184,
        "Precision": 0.9184,
        "Recall": 0.9184,
        "F1-score": 0.9183,
        "Training Time (seconds)": 4.89
    },
    {
        "Algorithm": "Random Forest",
        "Accuracy": 0.8838,
        "Precision": 0.8836,
        "Recall": 0.8838,
        "F1-score": 0.8832,
        "Training Time (seconds)": 110.92
    },
    {
        "Algorithm": "AdaBoost",
        "Accuracy": 0.6279,
        "Precision": 0.6673,
        "Recall": 0.6279,
        "F1-score": 0.6367,
        "Training Time (seconds)": 50.81
    },
    {
        "Algorithm": "Gradient Boosting",
        "Accuracy": 0.7211,
        "Precision": 0.7326,
        "Recall": 0.7211,
        "F1-score": 0.7214,
        "Training Time (seconds)": 2340.69
    },
    {
        "Algorithm": "XGBoost",
        "Accuracy": 0.8516,
        "Precision": 0.8528,
        "Recall": 0.8516,
        "F1-score": 0.8515,
        "Training Time (seconds)": 511.07
    }
]

df = pd.DataFrame(results)

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "model_comparison.csv"
df.to_csv(output_file, index=False)

print(df.to_string(index=False))
print(f"\nResults saved to: {output_file}")