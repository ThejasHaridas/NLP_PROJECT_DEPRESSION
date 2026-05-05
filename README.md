# Depression Detection from Social Media Text

An NLP project that classifies social media posts into depression severity levels using a systematic comparison of feature extraction methods and machine learning classifiers. Includes a Streamlit web application for real-time prediction.

## Overview

This project detects and classifies depression severity in user-generated text into three categories:

- **Not Depression** — No depression indicators present
- **Moderate** — Moderate depression indicators
- **Severe** — Severe depression indicators

The system was built by methodically comparing traditional NLP baselines against modern transformer-based approaches, with data augmentation applied to handle class imbalance.

## Project Structure

```
NLP_PROJECT_DEPRESSION/
├── app.py                        # Streamlit web application
├── requirements.txt              # Core dependencies
│
├── Data
│   ├── dev_with_labels.tsv       # Original labeled dataset
│   ├── Augdata.csv               # Augmented training dataset (5,576 samples)
│   └── test.csv                  # Test dataset
│
├── Trained Models
│   ├── classification_depression.pkl  # Primary production model
│   └── tfidf_dep.pkl                  # TF-IDF baseline model
│
├── Data Preparation
│   ├── csvmake.ipynb             # Data labeling and CSV preparation
│   └── Augmentation.ipynb        # Word2Vec-based data augmentation
│
├── Experiments — TF-IDF (1_*)
├── Experiments — CountVectorizer (2_*)
├── Experiments — Word2Vec Twitter (3_*)
├── Experiments — Word2Vec Google News 300d (4_*)
├── Experiments — BERT (5_bert*)
└── Experiments — Mental-BERT (6mentalbert*)
```

## Dataset

- **Source**: Social media depression forum posts
- **Labels**: `not depression`, `moderate`, `severe`
- **Class distribution** (after augmentation):
  - Moderate: 2,306 samples
  - Not depression: 1,830 samples
  - Severe: 1,440 samples

Data augmentation (`Augmentation.ipynb`) uses `nlpaug` with Word2Vec synonym substitution to balance the severe class.

## Approaches Explored

| # | Feature Extraction | Classifiers Tested |
|---|---|---|
| 1 | TF-IDF | Logistic Regression, Decision Tree, SVM, Random Forest |
| 2 | CountVectorizer | Logistic Regression, Decision Tree, SVM, Random Forest |
| 3 | Word2Vec (Twitter) | Logistic Regression, Decision Tree, SVM, Random Forest |
| 4 | Word2Vec (Google News 300d) | Decision Tree, SVM, Random Forest |
| 5 | BERT embeddings | Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost |
| 6 | Mental-BERT embeddings | Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost |

**Mental-BERT** (`mental/mental-bert-base-uncased`) is a BERT variant fine-tuned specifically on mental health text corpora.

## Results

| Approach | Overall Accuracy |
|---|---|
| TF-IDF + Logistic Regression | ~62% |
| BERT + classifier | ~68% |
| Mental-BERT + classifier | ~67% |

The BERT-based approach achieves the best overall performance, with notably strong detection of the severe class (93% precision, 80% recall).

## Web Application

The Streamlit app (`app.py`) provides a simple inference interface using the production model (`classification_depression.pkl`).

**Run the app:**

```bash
streamlit run app.py
```

Enter any text in the input field to receive a prediction: *not depressed*, *moderately depressed*, or *severely depressed*.

## Installation

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```
joblib
scikit-learn
streamlit
nltk
```

**Additional dependencies for running experiment notebooks:**

```
transformers
torch
gensim
nlpaug
numpy
pandas
```

## Usage

**Web app (inference only):**

```bash
streamlit run app.py
```

**Run an experiment notebook:**

Open any numbered notebook (e.g., `5_bertlogi.ipynb`) in Jupyter and run all cells. Notebooks are self-contained and load data from `Augdata.csv`.

## Ethical Note

This project involves sensitive mental health data and is intended for research purposes only. Predictions should not be used as a substitute for professional mental health assessment or diagnosis.
