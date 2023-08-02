##approval api
from flask import Flask, request, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from transformers import pipeline
import os
import sqlite3
import uuid
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize the login manager
login_manager = LoginManager()
login_manager.init_app(app)

# Define the path to the uploaded files directory
UPLOAD_DIR = 'uploaded_files'

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize the document-question-answering pipeline
pipeline_name = "impira/layoutlm-document-qa"
model = pipeline("document-question-answering", model=pipeline_name)

# Connect to the SQLite database
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Create the tables if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS files
             (id TEXT, file_path TEXT, receive_flag TEXT, status TEXT, questions TEXT, answers TEXT, user_id INTEGER)''')

c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, role TEXT)''')


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id


# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


# Register route for creating a new user
@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']

    c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
              (username, password, role))
    conn.commit()

    return jsonify({'message': 'User registered successfully'}), 200


# Login route for user authentication
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    c.execute("SELECT id, role FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()

    if user is None:
        return jsonify({'error': 'Invalid username or password'}), 401

    user_id, role = user

    # Create the User object and log the user in
    user_obj = User(user_id)
    login_user(user_obj)

    # Store the user role in the session
    session['role'] = role

    return jsonify({'message': 'Logged in successfully'}), 200


# Logout route
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200


# User-restricted route for uploading files and asking questions
@app.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Save the uploaded file to the specified directory
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    # Generate a UUID for the file
    file_id = str(uuid.uuid4())

    # Insert the file details into the database
    c.execute("INSERT INTO files (id, file_path, receive_flag, status, questions, answers, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (file_id, file_path, 'received', '', '[]', '[]', current_user.id))
    conn.commit()

    return jsonify({'message': 'File uploaded successfully', 'file_id': file_id}), 200


# User-restricted route for viewing uploaded files and corresponding question-answer pairs
@app.route('/files', methods=['GET'])
@login_required
def get_files():
    # Retrieve files belonging to the current user from the database
    c.execute("SELECT * FROM files WHERE user_id = ?", (current_user.id,))
    files = c.fetchall()

    # Format the files as a JSON response
    files_data = []
    for file in files:
        file_data = {
            'id': file[0],
            'file_path': file[1],
            'receive_flag': file[2],
            'status': file[3],
            'questions': json.loads(file[4]),
            'answers': json.loads(file[5])
        }
        files_data.append(file_data)

    return jsonify({'files': files_data}), 200


# Approver-restricted route for viewing uploaded files and corresponding question-answer pairs
@app.route('/approve/files', methods=['GET'])
@login_required
def get_pending_files():
    if session['role'] != 'approver':
        return jsonify({'error': 'Unauthorized access'}), 401

    # Retrieve files with 'received' status from the database
    c.execute("SELECT * FROM files WHERE status = 'received'")
    files = c.fetchall()

    # Format the files as a JSON response
    files_data = []
    for file in files:
        file_data = {
            'id': file[0],
            'file_path': file[1],
            'receive_flag': file[2],
            'status': file[3],
            'questions': json.loads(file[4]),
            'answers': json.loads(file[5])
        }
        files_data.append(file_data)

    return jsonify({'files': files_data}), 200


# Approver-restricted route for approving or denying a question-answer pair
@app.route('/approve/<qa_id>', methods=['PUT'])
@login_required
def approve_qa(qa_id):
    if session['role'] != 'approver':
        return jsonify({'error': 'Unauthorized access'}), 401

    # Retrieve the question-answer pair from the database
    c.execute("SELECT * FROM files WHERE id = ?", (qa_id,))
    qa_pair = c.fetchone()

    if qa_pair is None:
        return jsonify({'error': 'Question-Answer pair not found'}), 404

    # Update the status of the question-answer pair based on the request
    status = request.json.get('status')

    if status == 'approve':
        c.execute("UPDATE files SET status = ? WHERE id = ?", ('approved', qa_id))
        conn.commit()
    elif status == 'deny':
        c.execute("DELETE FROM files WHERE id = ?", (qa_id,))
        conn.commit()
    else:
        return jsonify({'error': 'Invalid status'}), 400

    return jsonify({'message': 'Question-Answer pair updated successfully'}), 200

