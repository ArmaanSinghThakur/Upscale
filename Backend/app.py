# backend/app.py

import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS 
from flask_sqlalchemy import SQLAlchemy # 👈 NEW IMPORT

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIST_DIR = os.path.join(BASE_DIR, 'dist')

app = Flask(
    __name__, 
    static_folder=FRONTEND_DIST_DIR, 
    static_url_path='/static'       
)
CORS(app) 

# Database Configuration for SQLite
# We'll create a database file named 'app.db' in the backend folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database object
db = SQLAlchemy(app) # 👈 NEW INITIALIZATION

# --- DATABASE MODELS (Step 3 will go here) ---
# backend/app.py (Insert after db = SQLAlchemy(app))

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128)) # To store a secure hash of the password

    def __repr__(self):
        return f'<User {self.username}>'
    
class FileHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # File details
    original_filename = db.Column(db.String(255), nullable=False)
    processed_filename = db.Column(db.String(255), nullable=True) # Will be NULL until processed
    
    # Status and Time
    status = db.Column(db.String(50), default='PENDING', nullable=False) # e.g., PENDING, PROCESSING, COMPLETED, FAILED
    uploaded_at = db.Column(db.DateTime, default=db.func.now())
    
    # Relationship to User (Foreign Key)
    # This links each file entry to a specific user's ID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationship property for SQLAlchemy
    # This allows us to access file history via user.files
    user = db.relationship('User', backref=db.backref('files', lazy=True))

    def __repr__(self):
        return f'<File {self.original_filename} - Status: {self.status}>'
# -----------------------
# --- API ROUTES ---
@app.route('/api/data')
def get_data():
    """Example API endpoint"""
    return jsonify({
        "message": "Hello from the Flask backend!",
        "status": "API is working with SQLite configured"
    })

# --- FRONTEND SERVING (PRODUCTION MODE) ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # ... (Keep the existing serving logic for production) ...
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(debug=True)