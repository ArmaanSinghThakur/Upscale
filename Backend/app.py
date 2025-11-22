import os
import sys
import uuid 
from werkzeug.utils import secure_filename 
from flask import Flask, jsonify, send_from_directory, request, abort 
from flask_cors import CORS 
from flask_sqlalchemy import SQLAlchemy 
from PIL import Image 

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from processor import process_image_for_upscale
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required 
from flask_bcrypt import Bcrypt

# --- UTILITY FUNCTION ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIST_DIR = os.path.join(BASE_DIR, 'dist')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads', 'originals')
# New folder for Avatars
AVATAR_FOLDER = os.path.join(BASE_DIR, 'uploads', 'avatars')

app = Flask(__name__, static_folder=FRONTEND_DIST_DIR, static_url_path='/static')

CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = '123456789' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AVATAR_FOLDER'] = AVATAR_FOLDER # 👈 NEW CONFIG
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Ensure avatar folder exists
if not os.path.exists(AVATAR_FOLDER):
    os.makedirs(AVATAR_FOLDER)

bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app) 
login_manager.session_protection = "strong"
login_manager.login_view = 'login' 
login_manager.login_message_category = "info"

db = SQLAlchemy(app) 

# --- DATABASE MODELS ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128)) 
    # 👇 NEW COLUMN: Avatar
    avatar_file = db.Column(db.String(120), nullable=True, default='default.png')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class FileHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    processed_filename = db.Column(db.String(255), nullable=True) 
    status = db.Column(db.String(50), default='PENDING', nullable=False) 
    uploaded_at = db.Column(db.DateTime, default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('files', lazy=True))

# --- API ROUTES ---

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/api/data')
def get_data():
    return jsonify({"message": "Backend Running"})

# --- AUTH ROUTES --- (Register, Login, Logout - Unchanged)
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username, email, password = data.get('username'), data.get('email'), data.get('password')
    if not username or not email or not password: return jsonify({"message": "Missing fields"}), 400
    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({"message": "User exists"}), 409
    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data.get('email')).first()
    if user and user.check_password(data.get('password')):
        login_user(user)
        return jsonify({"message": "Login successful", "username": user.username}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"}), 200

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True, 
            "username": current_user.username, 
            "id": current_user.id,
            # Return avatar URL helper
            "avatar": f"/api/avatar/{current_user.avatar_file}" if current_user.avatar_file else None
        }), 200
    else:
        return jsonify({"authenticated": False}), 200

# --- NEW PROFILE ROUTES ---

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Fetch user details and stats."""
    # Calculate stats
    upload_count = FileHistory.query.filter_by(user_id=current_user.id).count()
    
    return jsonify({
        "username": current_user.username,
        "email": current_user.email,
        "avatar": f"/api/avatar/{current_user.avatar_file}" if current_user.avatar_file else None,
        "stats": {
            "uploads": upload_count,
            "plan": "Free Tier" # Placeholder for future logic
        }
    }), 200

@app.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    """Update username or upload new avatar."""
    # Handle text data
    new_username = request.form.get('username')
    if new_username:
        current_user.username = new_username
    
    # Handle Avatar File
    if 'avatar' in request.files:
        file = request.files['avatar']
        if file and allowed_file(file.filename):
            # Secure name
            filename = secure_filename(file.filename)
            unique_name = f"user_{current_user.id}_{str(uuid.uuid4())[:8]}_{filename}"
            
            # Save
            save_path = os.path.join(app.config['AVATAR_FOLDER'], unique_name)
            file.save(save_path)
            
            # Update DB
            current_user.avatar_file = unique_name

    db.session.commit()
    return jsonify({"message": "Profile updated!", "username": current_user.username}), 200

@app.route('/api/avatar/<filename>')
def get_avatar(filename):
    """Serve avatar image."""
    return send_from_directory(app.config['AVATAR_FOLDER'], filename)


# --- FILE HANDLING ROUTES --- (Upload, History, Download, Delete - Unchanged)
@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files: return jsonify({"message": "No file"}), 400
    file = request.files['file']
    if not allowed_file(file.filename): return jsonify({"message": "Invalid type"}), 400
    
    original_secure = secure_filename(file.filename)
    unique_server = str(uuid.uuid4()) + os.path.splitext(original_secure)[1]
    
    try:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_server))
        new_file = FileHistory(original_filename=file.filename, user_id=current_user.id, processed_filename=unique_server)
        db.session.add(new_file)
        db.session.commit()
        
        if process_image_for_upscale(unique_server):
            new_file.status = 'COMPLETED'
            db.session.commit()
            return jsonify({"message": "Processed", "filename": unique_server.split('.')[0], "status": "completed"}), 201
        else:
            new_file.status = 'FAILED'
            db.session.commit()
            return jsonify({"message": "Processing failed", "status": "failed"}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/api/history', methods=['GET'])
@login_required
def get_user_history():
    files = FileHistory.query.filter_by(user_id=current_user.id).order_by(FileHistory.uploaded_at.desc()).all()
    return jsonify([{
        'id': f.id, 'original_filename': f.original_filename, 
        'unique_id': f.processed_filename.split('.')[0], 'status': f.status, 
        'date': f.uploaded_at.strftime('%Y-%m-%d')
    } for f in files]), 200

@app.route('/api/history/<int:file_id>', methods=['DELETE'])
@login_required
def delete_history_item(file_id):
    file = FileHistory.query.filter_by(id=file_id, user_id=current_user.id).first()
    if not file: return jsonify({"message": "Not found"}), 404
    try:
        # Try delete files (simplified for brevity)
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file.processed_filename)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file.processed_filename))
        db.session.delete(file)
        db.session.commit()
        return jsonify({"message": "Deleted"}), 200
    except: return jsonify({"message": "Error"}), 500

@app.route('/api/download/<string:base>', methods=['GET'])
def download_file(base):
    rec = FileHistory.query.filter(FileHistory.processed_filename.like(f'{base}%')).first()
    if not rec or rec.status != 'COMPLETED': return jsonify({"message": "Error"}), 404
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], '..', 'processed'), rec.processed_filename, as_attachment=True, download_name=f"upscaled_{rec.original_filename}")

@app.route('/api/view/<string:base>', methods=['GET'])
def view_file(base):
    rec = FileHistory.query.filter(FileHistory.processed_filename.like(f'{base}%')).first()
    if not rec: return jsonify({"message": "Error"}), 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], rec.processed_filename)

# --- SERVE FRONTEND ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True)