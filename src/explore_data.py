
import pandas as pd

# Load the saved datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Display dataset dimensions
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

# Check missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())

# Check duplicate rows
print("\nDuplicate training rows:", train_df.duplicated().sum())

# Display category distribution
print("\nCategory distribution:")
print(train_df["label"].value_counts().sort_index())

# Inspect sample articles
print("\nSample articles:")
for i in range(3):
    print(f"\nArticle {i + 1}:")
    print(train_df.loc[i, "text"])
    print("Label:", train_df.loc[i, "label"])