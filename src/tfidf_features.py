
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\\+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_text = train_df["text"].apply(clean_text)
test_text = test_df["text"].apply(clean_text)

# Extract labels
y_train = train_df["label"]
y_test = test_df["label"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# Fit only on training text
X_train = vectorizer.fit_transform(train_text)

# Transform test text using the same vocabulary
X_test = vectorizer.transform(test_text)

# Display results
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)
print("Number of vocabulary features:", len(vectorizer.vocabulary_))

# Display a few learned words
print("\nSample vocabulary:")
print(list(vectorizer.vocabulary_.keys())[:20])