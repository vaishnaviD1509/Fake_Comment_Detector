"""
Fake Comment Detection - ML backend.

Improvements over the original version:
  - Extra engineered features (exclamation count, caps ratio, digit ratio,
    keyword hits) in addition to has_link / comment_len.
  - TF-IDF now uses uni+bi-grams and drops ultra-rare terms.
  - Model selection: compares Linear SVM, RBF SVM, and Logistic Regression
    via stratified cross-validation and keeps the best one instead of
    hard-coding a single algorithm.
  - class_weight='balanced' so an imbalanced dataset doesn't bias predictions.
  - Reports precision/recall/F1 per class, not just accuracy (accuracy alone
    is misleading on imbalanced spam/ham style data).
"""
import re
import pickle

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --- Global Variables for state management ---
DATASET = None
X_train, X_test, y_train, y_test = None, None, None, None
VECTORIZER = None
SCALER = None
MODEL = None
MODEL_NAME = None
ACCURACY = None
REPORT_TEXT = None

# --- Configuration ---
TEXT_COLUMN = 'comment'
LABEL_COLUMN = 'label'
NUMERIC_FEATURES = ['has_link', 'comment_len', 'exclaim_count', 'caps_ratio',
                     'digit_ratio', 'keyword_hits']

SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'click', 'claim', 'subscribe', 'crypto',
    'cash', 'money', 'gift', 'prize', 'giveaway', 'link', 'bio', 'dm',
    'guarantee', 'guaranteed', 'urgent', 'verify', 'bonus', 'investment',
    'earn', 'rich', 'followers', 'coin',
]

_LEMMATIZER = None


def _ensure_nltk():
    for pkg in ('stopwords', 'wordnet', 'omw-1.4'):
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean_text(text):
    """Lowercase, strip non-letters, remove stopwords, lemmatize."""
    global _LEMMATIZER
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # drop urls before stripping punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()

    try:
        _ensure_nltk()
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        if _LEMMATIZER is None:
            _LEMMATIZER = WordNetLemmatizer()
        words = [_LEMMATIZER.lemmatize(w) for w in words]
    except LookupError:
        pass
    return ' '.join(words)


def _engineer_features(df):
    """Add engineered numeric features to a dataframe with a TEXT_COLUMN."""
    text = df[TEXT_COLUMN].astype(str)
    df = df.copy()
    df['comment_len'] = text.apply(len)
    df['has_link'] = text.apply(lambda x: 1 if ('http' in x or 'www' in x) else 0)
    df['exclaim_count'] = text.apply(lambda x: x.count('!'))
    df['caps_ratio'] = text.apply(
        lambda x: (sum(1 for c in x if c.isupper()) / len(x)) if len(x) > 0 else 0
    )
    df['digit_ratio'] = text.apply(
        lambda x: (sum(1 for c in x if c.isdigit()) / len(x)) if len(x) > 0 else 0
    )
    lowered = text.str.lower()
    df['keyword_hits'] = lowered.apply(
        lambda x: sum(1 for kw in SPAM_KEYWORDS if kw in x)
    )
    return df


# --- Main Functions corresponding to GUI/web actions ---

def read_dataset_func(file_path):
    """1. Reads the dataset from the selected CSV file."""
    global DATASET
    try:
        DATASET = pd.read_csv(file_path)
        DATASET.columns = [c.strip().lower() for c in DATASET.columns]
        if DATASET.empty or TEXT_COLUMN not in DATASET.columns or LABEL_COLUMN not in DATASET.columns:
            return (f"Error: Dataset is empty or missing required columns "
                     f"('{TEXT_COLUMN}' and '{LABEL_COLUMN}').")
        return f"Dataset loaded successfully. Rows: {len(DATASET)}"
    except FileNotFoundError:
        return f"Error: Dataset file not found at {file_path}"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


def pre_processing_func():
    """2. Cleans text data, engineers numerical features, and splits data."""
    global DATASET, X_train, X_test, y_train, y_test

    if DATASET is None:
        return "Error: Please read the dataset first."

    initial_rows = len(DATASET)

    DATASET = DATASET.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    DATASET = DATASET[DATASET[TEXT_COLUMN].astype(str).str.strip() != '']

    rows_dropped = initial_rows - len(DATASET)
    if len(DATASET) == 0:
        return "Error: No valid data remaining after removing missing values."

    if DATASET[LABEL_COLUMN].nunique() < 2:
        return "Error: Label column must have at least two unique classes (e.g., 0 and 1)."

    DATASET['cleaned_comment'] = DATASET[TEXT_COLUMN].astype(str).apply(clean_text)
    DATASET = DATASET[DATASET['cleaned_comment'].str.strip() != '']
    if len(DATASET) == 0:
        return "Error: No valid data remaining after text cleaning."

    DATASET = _engineer_features(DATASET)

    feature_columns = ['cleaned_comment'] + NUMERIC_FEATURES
    X = DATASET[feature_columns]
    y = DATASET[LABEL_COLUMN]

    try:
        y = pd.to_numeric(y, errors='coerce')
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices].astype(int)
        if len(y) == 0:
            return "Error: No valid numeric labels found in the dataset."
    except Exception as e:
        return f"Error converting labels to numeric: {str(e)}"

    # Guard against classes too small to stratify
    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None
    )

    msg = (f"Pre-processing complete. Training samples: {len(X_train)}, "
           f"Testing samples: {len(X_test)}")
    if rows_dropped:
        msg += f" (dropped {rows_dropped} invalid rows)"
    return msg


def classification_func():
    """3. Trains TF-IDF + numeric features, selects the best of several models."""
    global X_train, X_test, y_train, y_test, VECTORIZER, SCALER, MODEL, MODEL_NAME, ACCURACY, REPORT_TEXT

    if X_train is None or X_test is None:
        return "Error: Please complete pre-processing first."

    # --- Text features: TF-IDF with uni+bi-grams ---
    VECTORIZER = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X_train_text_vec = VECTORIZER.fit_transform(X_train['cleaned_comment'])
    X_test_text_vec = VECTORIZER.transform(X_test['cleaned_comment'])

    # --- Numeric features: scaled ---
    SCALER = StandardScaler(with_mean=False)  # keep sparse-compatible
    X_train_num = SCALER.fit_transform(X_train[NUMERIC_FEATURES].values)
    X_test_num = SCALER.transform(X_test[NUMERIC_FEATURES].values)

    X_train_vec = hstack([X_train_text_vec, csr_matrix(X_train_num)])
    X_test_vec = hstack([X_test_text_vec, csr_matrix(X_test_num)])

    # --- Model selection via cross-validation ---
    n_splits = min(5, int(y_train.value_counts().min())) if len(y_train.value_counts()) > 0 else 2
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    candidates = {
        'Linear SVM': SVC(kernel='linear', class_weight='balanced', random_state=42),
        'RBF SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42, gamma='scale'),
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42
        ),
    }

    best_name, best_score, best_model = None, -1, None
    cv_summary_lines = []
    try:
        for name, model in candidates.items():
            scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='f1_macro')
            mean_score = scores.mean()
            cv_summary_lines.append(f"  {name}: F1(macro)={mean_score:.3f}")
            if mean_score > best_score:
                best_score, best_name, best_model = mean_score, name, model
    except Exception as e:
        return f"Error during model selection: {str(e)}"

    try:
        best_model.fit(X_train_vec, y_train)
        MODEL, MODEL_NAME = best_model, best_name
    except Exception as e:
        return f"Error during model training: {str(e)}"

    try:
        y_pred = MODEL.predict(X_test_vec)
        ACCURACY = accuracy_score(y_test, y_pred)
        REPORT_TEXT = classification_report(y_test, y_pred, zero_division=0)
    except Exception as e:
        return f"Error during model evaluation: {str(e)}"

    cv_block = "\n".join(cv_summary_lines)
    return (
        f"Classification complete. Best model: {MODEL_NAME} "
        f"(chosen via {n_splits}-fold CV, F1={best_score:.3f}).\n"
        f"Test Accuracy: {ACCURACY * 100:.2f}%\n\n"
        f"Cross-validation comparison:\n{cv_block}\n\n"
        f"Test set report:\n{REPORT_TEXT}"
    )


def prediction_func(comment):
    """4. Predicts the label (Real/Fake) for a new comment."""
    global VECTORIZER, SCALER, MODEL

    if MODEL is None or VECTORIZER is None:
        return "Error: Please run 'Classification' first."

    cleaned_comment = clean_text(comment)
    if not cleaned_comment or cleaned_comment.strip() == '':
        return "Error: Comment is empty or contains no valid text after cleaning."

    try:
        comment_text_vec = VECTORIZER.transform([cleaned_comment])
    except Exception as e:
        return f"Error during vectorization: {str(e)}"

    row = pd.DataFrame([{TEXT_COLUMN: comment}])
    row = _engineer_features(row)
    comment_num = row[NUMERIC_FEATURES].values
    if SCALER is not None:
        comment_num = SCALER.transform(comment_num)

    comment_vec = hstack([comment_text_vec, csr_matrix(comment_num)])

    try:
        prediction = MODEL.predict(comment_vec)
        confidence = None
        if hasattr(MODEL, "decision_function"):
            try:
                score = MODEL.decision_function(comment_vec)
                confidence = float(1 / (1 + np.exp(-abs(score[0]))))  # rough calibration
            except Exception:
                confidence = None
        elif hasattr(MODEL, "predict_proba"):
            try:
                proba = MODEL.predict_proba(comment_vec)[0]
                confidence = float(max(proba))
            except Exception:
                confidence = None

        result = "FAKE COMMENT" if prediction[0] == 1 else "REAL COMMENT"
        conf_txt = f" (confidence ~{confidence*100:.0f}%)" if confidence is not None else ""
        return f"Prediction: **{result}**{conf_txt}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"


def save_model(model_path='model/svm_model.pkl', vectorizer_path='model/tfidf_vectorizer.pkl',
               scaler_path='model/scaler.pkl'):
    """Persist trained artifacts to disk."""
    if MODEL is None or VECTORIZER is None:
        return "Error: Nothing trained yet."
    with open(model_path, 'wb') as f:
        pickle.dump(MODEL, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(VECTORIZER, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(SCALER, f)
    return "Model artifacts saved."


def load_model(model_path='model/svm_model.pkl', vectorizer_path='model/tfidf_vectorizer.pkl',
               scaler_path='model/scaler.pkl'):
    """Load previously trained artifacts from disk, if present."""
    global MODEL, VECTORIZER, SCALER
    try:
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            VECTORIZER = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            SCALER = pickle.load(f)
        return True
    except Exception:
        return False