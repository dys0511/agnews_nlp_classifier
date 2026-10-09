
import pandas as pd
import re
import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load the dataset
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Clean the text
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
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 4. Train the Linear SVM model
model = LinearSVC(random_state=42)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# 5. Predict news categories
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\n--- Linear SVM Results ---")
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

# 7. Display the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["World", "Sports", "Business", "Sci/Tech"],
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Linear SVM - AG News Classification")
plt.tight_layout()
plt.show()