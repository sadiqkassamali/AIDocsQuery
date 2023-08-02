import os
import re
import json
import traceback
import csv
import urllib
from datetime import datetime
from threading import Lock
from urllib.error import URLError

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import textract
import PyPDF2
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import sqlite3
import threading
import uuid

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

from transformers import pipeline

app = Flask(__name__)
lock = Lock()

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DB_FILE = 'file_database.db'
PAGE_SIZE = 10  # Number of files per page

// set up rate limiter: maximum of five requests per minute
var RateLimit = require('express-rate-limit');
var limiter = RateLimit({
  windowMs: 1*60*1000, // 1 minute
  max: 200
});

// apply rate limiter to all requests
app.use(limiter);

def create_table():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            model TEXT,
            dataset_path TEXT,
            extracted_text TEXT
        )
    ''')
    conn.close()


def preprocess_image(image_path):
    try:
        with Image.open(image_path) as image:
            # Convert PNG to RGB

            # Get image dimensions
            width, height = image.size

            # Convert to grayscale
            grayscale_image = image.convert("L")

            # Enhance image contrast
            enhanced_image = ImageEnhance.Contrast(grayscale_image).enhance(2.0)

            # Apply image filters
            filtered_image = enhanced_image.filter(ImageFilter.SHARPEN)

            return filtered_image, width, height
    except IOError:
        print("Error opening image file:", image_path)
        return None, None, None


def extract_text_from_image(file_path):
    try:
        # Preprocess the image
        processed_image, width, height = preprocess_image(file_path)

        if not processed_image:
            return None

        # Use pytesseract to perform OCR on the processed image
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng')

        return extracted_text.strip(), width, height
    except IOError:
        print("Unable to open image file:", file_path)
        return None, None, None
    except Exception as e:
        traceback.print_exc()
        return None, None, None


def extract_text_from_pdf(file_path):
    try:
        if isinstance(file_path, str):
            if os.path.isfile(file_path):
                # Open the PDF file in binary mode
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    extracted_text = ""
                    for page in pdf_reader.pages:
                        extracted_text += page.extract_text()

                return extracted_text.strip()
            else:
                raise FileNotFoundError("PDF file not found:", file_path)
        else:
            raise TypeError("Invalid file path provided.")
    except Exception as e:
        traceback.print_exc()
        return None


def extract_text_from_doc(file_path):
    try:
        # Use textract to extract text from DOC
        extracted_text = textract.process(file_path).decode('utf-8')

        return extracted_text.strip()
    except IOError:
        print("Unable to open DOC file:", file_path)
        return None
    except Exception as e:
        traceback.print_exc()
        return None


def extract_text_from_csv(file_path):
    try:
        extracted_text = ""
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                extracted_text += ' '.join(row) + '\n'

        return extracted_text.strip()
    except IOError:
        print("Unable to open CSV file:", file_path)
        return None
    except Exception as e:
        traceback.print_exc()
        return None


def extract_text_from_xls(file_path):
    try:
        extracted_text = ""
        data_frame = pd.read_excel(file_path)
        for column in data_frame.columns:
            extracted_text += ' '.join([str(cell) for cell in data_frame[column]]) + '\n'

        return extracted_text.strip()
    except IOError:
        print("Unable to open XLS file:", file_path)
        return None
    except Exception as e:
        traceback.print_exc()
        return None


def post_process_text(text):
    if text is None:
        return None

    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()


def visiontext(cleaned_text):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    if cleaned_text is not None:
        sequence_to_classify = cleaned_text
    else:
        sequence_to_classify = "unknown"
    candidate_labels = ['License', 'Card', 'CONTRACT', 'AGREEMENT', 'Identification Card', 'legal']
    predictions = classifier(sequence_to_classify, candidate_labels)
    predicted_class = predictions['labels'][0]
    predicted_score = predictions['scores'][0]

    result = {
        "predicted_class": predicted_class,
        "predicted_score": predicted_score
    }

    return result


def vision(image):
    dataset = load_dataset("aharley/rvl_cdip")

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])

    return predicted_label


def is_url_reachable(url):
    try:
        response = urllib.request.urlopen(url)
        return True
    except URLError:
        return False


def extract_text(file_paths):
    extracted_text = []

    for file_path in file_paths:
        file_metadata = {}

        # Get file information
        file_name = os.path.basename(file_path)
        file_size = None
        file_modified = None
        file_extension = os.path.splitext(file_path)[1].lower()

        file_metadata["name"] = file_name
        file_metadata["size"] = file_size
        file_metadata["last_modified"] = file_modified

        if file_extension in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'):
            # Handle local image file paths
            if os.path.isfile(file_path):
                extracted_image_text, width, height = extract_text_from_image(file_path)
            # Handle image file paths as URLs
            else:
                try:
                    file_path = urllib.parse.unquote(file_path)  # Decode URL-encoded path

                    # Handle Windows file paths
                    if os.name == 'nt':
                        file_path = file_path.replace('/', '\\')

                    if not is_url_reachable(file_path):
                        print("URL is not reachable:", file_path)
                        continue

                    with urllib.request.urlopen(file_path) as url_file:
                        image_data = url_file.read()
                        temp_file_path = os.path.join(UPLOAD_FOLDER, file_name)
                        with open(temp_file_path, 'wb') as temp_file:
                            temp_file.write(image_data)
                    extracted_image_text, width, height = extract_text_from_image(temp_file_path)
                    os.remove(temp_file_path)
                except Exception as e:
                    traceback.print_exc()
                    extracted_image_text = None

            if extracted_image_text is not None:
                cleaned_text = post_process_text(extracted_image_text)
                vision_results = visiontext(cleaned_text)
                custom_vision_result = vision(Image.open(file_path))

                extracted_text.append({
                    "file_path": file_path,
                    "image_text": cleaned_text,
                    "vision_results": vision_results,
                    "custom_vision_result": custom_vision_result,
                    "metadata": file_metadata,
                    "width": width,
                    "height": height
                })
        # Handle other file types similarly
        # ...

    return extracted_text


def update_status(file_id, status):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE file_entries
        SET status = ?
        WHERE id = ?
    ''', (status, file_id))
    conn.commit()
    conn.close()


def update_extracted_text(file_id, extracted_text):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE file_entries
        SET extracted_text = ?
        WHERE id = ?
    ''', (extracted_text, file_id))
    conn.commit()
    conn.close()


def delete_file_from_db(file_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM file_entries
        WHERE id = ?
    ''', (file_id,))
    conn.commit()
    conn.close()


def delete_model_and_dataset(file_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT model, dataset_path
        FROM file_entries
        WHERE id = ?
    ''', (file_id,))
    entry = cursor.fetchone()

    if entry:
        # Delete the model file from cache
        model_path = entry[0]
        if model_path:
            os.remove(model_path)

        # Delete the dataset cache directory
        dataset_path = entry[1]
        if dataset_path:
            os.remove(dataset_path)

    # Clear the model and dataset paths in the database entry
    cursor.execute('''
        UPDATE file_entries
        SET model = NULL,
            dataset_path = NULL
        WHERE id = ?
    ''', (file_id,))
    conn.commit()
    conn.close()


def insert_file(file_path):
    # Insert file data into the SQLite database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Generate a UUID for the file entry
    file_uuid = str(uuid.uuid4())

    # Insert the file entry into the database
    cursor.execute('''
        INSERT INTO file_entries (uuid, file_path)
        VALUES (?, ?)
    ''', (file_uuid, file_path))
    file_id = cursor.lastrowid

    conn.commit()
    conn.close()

    # Start a new thread to process the file
    thread = threading.Thread(target=process_file, args=(file_id, file_path))
    thread.start()

    return file_uuid


def get_all_files_from_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT uuid, file_path, status
        FROM file_entries
    ''')
    files = cursor.fetchall()
    conn.close()

    response = []
    for file in files:
        file_info = {
            'uuid': file[0],
            'file_path': file[1],
            'status': file[2]
        }
        response.append(file_info)

    return response


def get_file_from_db(file_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT uuid, file_path, status
        FROM file_entries
        WHERE uuid = ?
    ''', (file_id,))
    file = cursor.fetchone()
    conn.close()

    if file:
        file_info = {
            'uuid': file[0],
            'file_path': file[1],
            'status': file[2]
        }
        return file_info
    else:
        return None


def process_file(file_id, file_path):
    try:
        # Set the status to 'in process'
        update_status(file_id, 'in process')

        # Extract text from the file
        extracted_text = extract_text(file_path)

        # Update the extracted text in the database
        update_extracted_text(file_id, json.dumps(extracted_text))

        # Set the status to 'done'
        update_status(file_id, 'done')
    except Exception as e:
        traceback.print_exc()
        # Set the status to 'error'
        update_status(file_id, 'error')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Here, you can implement the logic to validate the user's credentials.
        # For simplicity, let's assume the username is 'admin' and the password is 'password'.
        if username == 'admin' and password == 'password':
            # In a real application, you would set a session to mark the user as logged in.
            return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        file_uuid = insert_file(file_path)
        return jsonify({'message': 'File uploaded successfully.', 'file_uuid': file_uuid}), 200


@app.route('/extract', methods=['POST'])
def extract_text_api():
    file_uuids = request.json.get('file_uuids', [])

    if not file_uuids:
        return jsonify({'error': 'No file UUIDs provided.'}), 400

    extracted_text = []

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for file_uuid in file_uuids:
        cursor.execute('''
            SELECT extracted_text
            FROM file_entries
            WHERE uuid = ?
        ''', (file_uuid,))
        entry = cursor.fetchone()

        if entry:
            extracted_text_json = entry[0]
            extracted_text.append(json.loads(extracted_text_json))

    conn.close()

    if extracted_text:
        json_output = json.dumps(extracted_text, indent=4)
        return json_output, 200
    else:
        return jsonify({'message': 'No text extracted for the provided file UUIDs.'}), 200


@app.route('/files/all', methods=['GET'])
def get_all_files():
    try:
        files = get_all_files_from_db()
        return jsonify(files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/files/<file_uuid>', methods=['GET'])
def get_file(file_uuid):
    try:
        file_data = get_file_from_db(file_uuid)
        if file_data:
            return jsonify(file_data), 200
        else:
            return jsonify({'message': 'File not found.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload-file', methods=['POST'])
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
    c.execute("INSERT INTO files (id, file_path, receive_flag, status, question, answer) VALUES (?, ?, ?, ?, ?, ?)",
              (file_id, file_path, 'received', '', '', ''))
    conn.commit()
   
    return jsonify({'message': 'File uploaded successfully', 'file_id': file_id}), 200


@app.route('/query-with-file', methods=['POST'])
def query_with_file():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
   
    file = request.files['file']
    question = request.form.get('question')
   
    # Save the uploaded file to the specified directory
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)
   
    # Perform question-answering using the uploaded file
    result = model({
        'question': question,
        'context': file_path
    })
   
    # Retrieve the answer from the result
    answer = result['answer']
   
    # Generate a UUID for the question-answer pair
    qa_id = str(uuid.uuid4())
   
    # Insert the file details and question-answer pair into the database
    c.execute("UPDATE files SET status = ?, question = ?, answer = ? WHERE file_path = ?",
              ('done', question, answer, file_path))
    conn.commit()
   
    return jsonify({'answer': answer, 'qa_id': qa_id}), 200


@app.route('/files', methods=['GET'])
def get_files():
    # Retrieve all the files from the database
    c.execute("SELECT * FROM files")
    files = c.fetchall()
   
    # Format the files as a JSON response
    files_data = []
    for file in files:
        file_data = {
            'id': file[0],
            'file_path': file[1],
            'receive_flag': file[2],
            'status': file[3],
            'question': file[4],
            'answer': file[5]
        }
        files_data.append(file_data)
   
    return jsonify({'files': files_data}), 200


@app.route('/files/<file_id>', methods=['GET'])
def get_file(file_id):
    # Retrieve the file with the specified ID from the database
    c.execute("SELECT * FROM files WHERE id = ?", (file_id,))
    file = c.fetchone()
   
    if file is None:
        return jsonify({'error': 'File not found'}), 404
   
    # Format the file as a JSON response
    file_data = {
        'id': file[0],
        'file_path': file[1],
        'receive_flag': file[2],
        'status': file[3],
        'question': file[4],
        'answer': file[5]
    }
   
    return jsonify({'file': file_data}), 200

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

if __name__ == '__main__':
    create_table()
    app.run(threaded=True)
