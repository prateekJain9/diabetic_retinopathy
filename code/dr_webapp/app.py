import os
import io
import base64
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify)
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# ── TF import with version-safe fallback ──
TF_AVAILABLE = False
tf = None
tf_load_model = None
try:
    import tensorflow as _tf
    from tensorflow.keras.models import load_model as _load_model
    tf = _tf
    tf_load_model = _load_model
    TF_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────
#  App Config
# ─────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR  = os.path.join(BASE_DIR, 'instance')
DB_PATH       = os.path.join(INSTANCE_DIR, 'dr_app.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH    = os.path.join(BASE_DIR, 'dr_fedavg_model.h5')
ALLOWED_EXTS  = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
IMG_SIZE      = (224, 224)
CLASS_NAMES   = ['DR', 'No_DR']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ── Create required directories (fixes "unable to open database" on Windows) ──
os.makedirs(INSTANCE_DIR,  exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────
#  Model loading
#
#  The model was saved with Keras 3.x which stores DTypePolicy
#  objects and batch_shape/optional in InputLayer configs.
#  We patch the HDF5 config JSON in a temp file before loading.
# ─────────────────────────────────────────

def _load_model_compat(path):
    """Patch Keras 3.x H5 model config to load in any TF version."""
    import h5py, json, shutil, tempfile, os as _os

    with h5py.File(path, 'r') as f:
        cfg_raw = f.attrs['model_config']
        if isinstance(cfg_raw, bytes):
            cfg_raw = cfg_raw.decode('utf-8')
    cfg = json.loads(cfg_raw)

    # ✅ Properly indented patch function
    def patch(node):
        if isinstance(node, dict):

            # Remove quantization_config
            if 'quantization_config' in node:
                node.pop('quantization_config', None)

            # Fix dtype policy
            if node.get('class_name') == 'DTypePolicy':
                return node.get('config', {}).get('name', 'float32')

            # Fix InputLayer
            if node.get('class_name') == 'InputLayer':
                c = node.get('config', {})
                bs = c.pop('batch_shape', None)
                c.pop('optional', None)
                c.pop('sparse', None)
                c.pop('ragged', None)

                if bs is not None and 'batch_input_shape' not in c:
                    c['batch_input_shape'] = bs

            # Recursive fix
            for k in list(node.keys()):
                r = patch(node[k])
                if r is not None:
                    node[k] = r

        elif isinstance(node, list):
            for i in range(len(node)):
                r = patch(node[i])
                if r is not None:
                    node[i] = r

        return None

    # Apply patch
    patch(cfg)

    # Create temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp.close()
    shutil.copy2(path, tmp.name)

    try:
        with h5py.File(tmp.name, 'r+') as f:
            f.attrs['model_config'] = json.dumps(cfg)

        try:
            m = tf_load_model(tmp.name, compile=False)
            return m
        except Exception as e:
            print("[ERROR] Actual model loading error:", e)
            return None

    finally:
        _os.unlink(tmp.name)



model = None
if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = _load_model_compat(MODEL_PATH)

        if model is not None:
            print(f"[INFO] Model loaded: {model.input_shape} -> {model.output_shape}")
        else:
            print("[ERROR] Model is None after loading")

    except Exception as e:
        print(f"[WARN] Model load failed: {e}")
        model = None

elif not os.path.exists(MODEL_PATH):
    print(f"[WARN] Model file not found at: {MODEL_PATH}")

elif not TF_AVAILABLE:
    print("[WARN] TensorFlow not installed - prediction disabled.")
# ─────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            email     TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            created   TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            filename    TEXT,
            label       TEXT,
            confidence  REAL,
            dr_prob     REAL,
            no_dr_prob  REAL,
            created     TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

init_db()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ─────────────────────────────────────────
#  Auth decorator
# ─────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────
#  Prediction helper
# ─────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

def predict_image(path):
    img   = Image.open(path).convert('RGB').resize(IMG_SIZE)
    arr   = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, 0)
    probs = model.predict(batch, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, float(probs[0]) * 100, float(probs[1]) * 100

# ─────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

# ── Auth ──

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name  = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        pw    = request.form.get('password', '')
        pw2   = request.form.get('confirm_password', '')

        if not all([name, email, pw, pw2]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        if pw != pw2:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        if len(pw) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')

        conn = get_db()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?,?,?)',
                         (name, email, hash_pw(pw)))
            conn.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'danger')
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        pw    = request.form.get('password', '')

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email=? AND password=?',
                            (email, hash_pw(pw))).fetchone()
        conn.close()

        if user:
            session['user_id']   = user['id']
            session['user_name'] = user['name']
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# ── Dashboard ──

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    recent = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 5',
        (session['user_id'],)
    ).fetchall()
    total = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id=?',
                         (session['user_id'],)).fetchone()[0]
    dr_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id=? AND label='DR'",
        (session['user_id'],)).fetchone()[0]
    conn.close()
    return render_template('dashboard.html', recent=recent, total=total, dr_count=dr_count)

# ── Prediction ──

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file selected.', 'danger')
            return render_template('predict.html', result=None)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return render_template('predict.html', result=None)

        if not allowed_file(file.filename):
            flash('Only image files are allowed (PNG, JPG, JPEG, BMP, TIFF).', 'danger')
            return render_template('predict.html', result=None)

        filename = secure_filename(file.filename)
        ts_name  = f"{session['user_id']}_{int(datetime.now().timestamp())}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, ts_name)
        file.save(filepath)

        if model is None:
            flash('Model not loaded. Please ensure dr_fedavg_model.h5 is in the app folder.', 'danger')
            return render_template('predict.html', result=None)

        try:
            label, conf, dr_p, no_dr_p = predict_image(filepath)

            conn = get_db()
            conn.execute(
                'INSERT INTO predictions (user_id,filename,label,confidence,dr_prob,no_dr_prob) VALUES (?,?,?,?,?,?)',
                (session['user_id'], ts_name, label, conf, dr_p, no_dr_p)
            )
            conn.commit()
            conn.close()

            with open(filepath, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            ext  = filename.rsplit('.', 1)[1].lower()
            mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'

            result = {
                'label':      label,
                'confidence': round(conf, 2),
                'dr_prob':    round(dr_p, 2),
                'no_dr_prob': round(no_dr_p, 2),
                'image_data': f'data:{mime};base64,{img_b64}'
            }
        except Exception as e:
            flash(f'Prediction error: {e}', 'danger')

    return render_template('predict.html', result=result)

# ── Analytics ──

@app.route('/analytics')
@login_required
def analytics():
    conn = get_db()
    uid  = session['user_id']

    all_preds = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC', (uid,)
    ).fetchall()

    total    = len(all_preds)
    dr_count = sum(1 for p in all_preds if p['label'] == 'DR')
    no_dr    = total - dr_count
    avg_conf = round(sum(p['confidence'] for p in all_preds) / total, 2) if total else 0

    days = {}
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime('%b %d')
        days[day] = 0
    for p in all_preds:
        try:
            day = datetime.fromisoformat(p['created']).strftime('%b %d')
            if day in days:
                days[day] += 1
        except Exception:
            pass

    conn.close()

    return render_template('analytics.html',
        predictions=all_preds,
        total=total,
        dr_count=dr_count,
        no_dr_count=no_dr,
        avg_conf=avg_conf,
        days=list(days.keys()),
        day_counts=list(days.values())
    )

# ── API history ──

@app.route('/api/history')
@login_required
def api_history():
    conn = get_db()
    rows = conn.execute(
        'SELECT id,label,confidence,dr_prob,no_dr_prob,created FROM predictions '
        'WHERE user_id=? ORDER BY id DESC LIMIT 20',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


if __name__ == '__main__':
    app.run(debug=True, port=5000)