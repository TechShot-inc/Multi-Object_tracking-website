from flask import Flask, render_template
import os

app = Flask(__name__)

# Directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
APP_DIR = BASE_DIR
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'Uploads')
app.config['RESULTS_FOLDER'] = os.path.join(PROJECT_ROOT, 'results')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Logging paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
print(f"RESULTS_FOLDER: {app.config['RESULTS_FOLDER']}")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Import routes
from .realtime import *
from .video import *

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video')
def video():
    """Render the video tracking page."""
    return render_template('video.html')

@app.route('/realtime')
def realtime():
    """Render the real-time tracking page."""
    return render_template('realtime.html')