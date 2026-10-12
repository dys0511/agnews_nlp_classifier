
import pandas as pd
import re
import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

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

# 3. TF-IDF: use fewer features to reduce memory
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 1),
    sublinear_tf=True,
    dtype="float32"
)

X_train_sparse = vectorizer.fit_transform(X_train_text)
X_test_sparse = vectorizer.transform(X_test_text)

print("TF-IDF features created.")

# 4. Convert to dense float32 arrays
X_train = X_train_sparse.toarray()
X_test = X_test_sparse.toarray()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 5. Train Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=30,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("\n--- Gradient Boosting Results ---")
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
    y_test, y_pred,
    target_names=["World", "Sports", "Business", "Sci/Tech"]
))

# 8. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["World", "Sports", "Business", "Sci/Tech"],
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Gradient Boosting - AG News Classification")
plt.tight_layout()
plt.show()
