# Fake Comment Detector — Web App

A Flask web version of the original Tkinter desktop app, with an improved
ML pipeline.

## What changed from the original

**Dataset**
- The original `comment.csv` had only 15 rows, one of them malformed
  (an unquoted comma inside the text broke CSV parsing).
- `data/generate_dataset.py` synthesizes a larger, properly-quoted,
  balanced dataset (`data/comment.csv`, ~290 rows: genuine comments vs.
  spam/scam/engagement-bait comments) so the model has enough signal to
  learn from. Swap in your own CSV any time (needs `comment` and `label`
  columns) — via the web UI's upload button.

**Model (`ml_processor.py`)**
- More engineered features: exclamation count, capital-letter ratio,
  digit ratio, and a spam-keyword hit count, in addition to comment
  length and link presence.
- TF-IDF now uses uni+bigrams instead of unigrams only.
- Numeric features are scaled (`StandardScaler`) before being combined
  with the TF-IDF matrix.
- Instead of hard-coding a linear SVM, `classification_func()` runs
  stratified cross-validation over Linear SVM / RBF SVM / Logistic
  Regression and keeps whichever scores best on macro-F1, with
  `class_weight='balanced'` so class imbalance doesn't skew predictions.
- Reports precision/recall/F1 per class (not just accuracy) since
  accuracy alone is misleading on imbalanced spam/ham-style data.
- Predictions now include a rough confidence percentage.

**Interface**
- Same 4-step workflow (load → preprocess → train → predict) and the
  same login (`admin` / `admin`), now as a Flask web app instead of a
  Tkinter desktop window, so it runs in a browser and could be deployed.
- Dark/light theme toggle (client-side, no page reload).
- A trained model is auto-saved to `model/` after training and reloaded
  on server startup, so you don't need to retrain every time you restart
  the app — the bundled `model/` already has an initial trained model
  from the sample dataset.

## Setup

```bash
python3 -m venv venv
on Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Then open **http://localhost:5000** in your browser.

Log in with `admin` / `admin`, click **Start detection**, then either
upload your own CSV or click **Use sample dataset**, and step through
Preprocess → Train → Analyze a comment.

## Regenerating the sample dataset

```bash
cd data
python3 generate_dataset.py
```

## Notes / limitations

- This keeps the original app's design of a single shared model state
  (like the desktop app's global variables) rather than a per-user
  database — fine for local/single-user use, but if you deploy this for
  multiple concurrent users you'll want to move dataset/model state into
  a database or per-session storage instead of module-level globals.
- The bundled dataset is synthetic and meant to make the pipeline
  actually learnable; for production use, replace it with a real,
  labeled dataset of comments from your own platform.
- The login is hard-coded (`admin`/`admin`) exactly as it was in the
  original desktop app — replace with real authentication before
  deploying this anywhere accessible to others.
