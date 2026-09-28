import os
import threading
import webbrowser
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

import ml_processor as mp

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, 'data', 'uploads')
MODEL_DIR = os.path.join(APP_DIR, 'model')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'dev-secret-key-change-in-production'

# Try to load a previously trained model on startup so "Analyze" works
# immediately without forcing a retrain every time the server restarts.
_PRELOADED = mp.load_model(
    model_path=os.path.join(MODEL_DIR, 'svm_model.pkl'),
    vectorizer_path=os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'),
    scaler_path=os.path.join(MODEL_DIR, 'scaler.pkl'),
)


def login_required(view):
    def wrapped(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        error = 'Invalid username or password'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session.get('username', 'admin'))


@app.route('/detect')
@login_required
def detect():
    return render_template(
        'detect.html',
        model_ready=(mp.MODEL is not None),
        preloaded=_PRELOADED,
    )


# ---------------------------------------------------------------------------
# API endpoints used by detect.html (AJAX)
# ---------------------------------------------------------------------------

@app.route('/api/status')
@login_required
def api_status():
    return jsonify({
        'dataset_loaded': mp.DATASET is not None,
        'preprocessed': mp.X_train is not None,
        'trained': mp.MODEL is not None,
        'model_name': mp.MODEL_NAME,
        'accuracy': mp.ACCURACY,
    })


@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'ok': False, 'message': 'No file selected.'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'ok': False, 'message': 'Please upload a .csv file.'}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)
    status = mp.read_dataset_func(save_path)
    ok = status.lower().startswith('dataset loaded')
    return jsonify({'ok': ok, 'message': status})


@app.route('/api/use-sample', methods=['POST'])
@login_required
def api_use_sample():
    sample_path = os.path.join(APP_DIR, 'data', 'comment.csv')
    status = mp.read_dataset_func(sample_path)
    ok = status.lower().startswith('dataset loaded')
    return jsonify({'ok': ok, 'message': status})


@app.route('/api/preprocess', methods=['POST'])
@login_required
def api_preprocess():
    status = mp.pre_processing_func()
    ok = status.lower().startswith('pre-processing complete')
    return jsonify({'ok': ok, 'message': status})


@app.route('/api/train', methods=['POST'])
@login_required
def api_train():
    status = mp.classification_func()
    ok = status.lower().startswith('classification complete')
    if ok:
        mp.save_model(
            model_path=os.path.join(MODEL_DIR, 'svm_model.pkl'),
            vectorizer_path=os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'),
            scaler_path=os.path.join(MODEL_DIR, 'scaler.pkl'),
        )
    return jsonify({'ok': ok, 'message': status})


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    comment = (data.get('comment') or '').strip()
    if not comment:
        return jsonify({'ok': False, 'message': 'Please enter a comment.'}), 400

    status = mp.prediction_func(comment)
    ok = status.startswith('Prediction:')
    is_fake = 'FAKE' in status
    return jsonify({'ok': ok, 'message': status.replace('**', ''), 'is_fake': is_fake})


if __name__ == '__main__':
    # Auto-open the browser once the server starts. Guarded against Flask's
    # debug-mode reloader (which re-executes this file in a child process)
    # so the tab doesn't open twice.
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Timer(1.0, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(debug=True, host='0.0.0.0', port=5000)