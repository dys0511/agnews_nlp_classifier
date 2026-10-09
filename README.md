# AG News Classification Using Machine Learning

## Project Overview

This project uses Natural Language Processing (NLP) and machine learning algorithms to classify news articles into four categories: World, Sports, Business and Sci/Tech.

The project uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features and compares five machine learning algorithms.

## Dataset

* **Dataset:** AG News
* **Training samples:** 120,000
* **Testing samples:** 7,600
* **Categories:** World, Sports, Business and Sci/Tech
* **Source:** [AG News Dataset on Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news)

## Algorithms Implemented

1. Linear Support Vector Machine (SVM)
2. Random Forest Classifier
3. AdaBoost Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

## Methodology

1. Load the AG News dataset.
2. Clean text by converting it to lowercase and removing unwanted backslashes and extra whitespace.
3. Convert text into TF-IDF features.
4. Train the five machine learning models.
5. Evaluate the models using accuracy, macro precision, macro recall, macro F1-score and training time.
6. Compare the results using a CSV file and visualizations.

## Results

| Algorithm         | Accuracy | Macro F1-score |
| ----------------- | -------: | -------------: |
| Linear SVM        |   91.84% |         91.83% |
| Random Forest     |   88.38% |         88.32% |
| XGBoost           |   85.16% |         85.15% |
| Gradient Boosting |   72.11% |         72.14% |
| AdaBoost          |   62.79% |         63.67% |

**Best-performing model:** Linear SVM achieved the highest accuracy of 91.84% in the experiments.

**Experimental limitation:** Gradient Boosting was trained on a stratified sample of 20,000 articles using 2,000 unigram features because of computational limitations. The other models used 120,000 training articles and 20,000 unigram-and-bigram features, so the results are not a perfectly controlled comparison.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* TF-IDF Vectorization

## Project Structure

```text
ag-news-classifier/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── tfidf_features.py
│   ├── linear_svm.py
│   ├── random_forest.py
│   ├── adaboost.py
│   ├── gradient_boost.py
│   ├── xgboost_model.py
│   ├── save_results.py
│   └── compare_models.py
├── results/
│   ├── model_comparison.csv
│   └── graphs/
├── requirements.txt
└── README.md
```

*Note: Adjust the filenames above to match the actual files in your repository.*

## How to Run

1. Install Python 3.10 or later.
2. Clone or download this repository.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Ensure the dataset files are available in the `data/` folder.
6. Run an algorithm from the project root, for example:

```bash
python src/linear_svm.py
```

To generate the results table and comparison graphs:

```bash
python src/save_results.py
python src/compare_models.py
```

## Conclusion

The experiments demonstrate that TF-IDF features combined with machine learning algorithms can effectively classify news articles. Linear SVM achieved the best observed accuracy and the shortest training time among the tested models. The results also highlight the importance of computational efficiency and consistent experimental settings when comparing classification algorithms.
