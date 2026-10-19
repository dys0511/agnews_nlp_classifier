
import pandas as pd
import re
import time 
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# 1. Load dataset
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\\+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

X_train_text = train_df["text"].apply(clean_text)
X_test_text = test_df["text"].apply(clean_text)

y_train = train_df["label"]
y_test = test_df["label"]

# 3. Convert text into TF-IDF features
# Keep features sparse to reduce memory usage
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype="float32"
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# 4. Train XGBoost
model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# 5. Predict categories
y_pred = model.predict(X_test)

# 6. Evaluate model
print("\n--- XGBoost Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Macro Precision:", round(
    precision_score(y_test, y_pred, average="macro"), 4
))
print("Macro Recall:", round(
    recall_score(y_test, y_pred, average="macro"), 4
))
print("Macro F1-score:", round(
    f1_score(y_test, y_pred, average="macro"), 4
))
print("Training time:", round(training_time, 2), "seconds")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["World", "Sports", "Business", "Sci/Tech"]
))

# 7. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["World", "Sports", "Business", "Sci/Tech"],
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("XGBoost - AG News Classification")
plt.tight_layout()
plt.show()
