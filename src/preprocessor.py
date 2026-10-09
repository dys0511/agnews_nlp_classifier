
import pandas as pd
import re

# Load the original datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# Text cleaning function
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Replace literal backslashes with spaces
    text = re.sub(r"\\+", " ", text)

    # Remove extra whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning to both datasets
train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)


# Display original-style cleaned samples
print("Cleaned training samples:\n")

for i in range(3):
    print(f"Article {i + 1}:")
    print(train_df.loc[i, "text"])
    print("Label:", train_df.loc[i, "label"])
    print()

print("Text cleaning completed!")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))