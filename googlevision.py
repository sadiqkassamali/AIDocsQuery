import os
import re
import json
import traceback
import csv

import urllib.parse
import urllib.request
from urllib.error import URLError
from threading import Lock

import textract
import PyPDF2
import pandas as pd

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
lock = Lock()

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
# Define paths for cached models
CACHE_DIR = 'model_cache'
RESNET_CACHE_DIR = os.path.join(CACHE_DIR, 'resnet')
BART_CACHE_DIR = os.path.join(CACHE_DIR, 'bart')

# Create cache directories if they don't exist
os.makedirs(RESNET_CACHE_DIR, exist_ok=True)
os.makedirs(BART_CACHE_DIR, exist_ok=True)

CACHE_DIR = 'dataset_cache'  # Directory to store the cached dataset

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Define the name and parameters of your dataset
CACHE_DIR = 'dataset_cache'  # Directory to store the cached dataset

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Check if the dataset is already cached
if not os.path.exists(os.path.join(CACHE_DIR, 'aharley/rvl_cdip')):
    # Load and cache the dataset
    dataset = load_dataset('aharley/rvl_cdip')
    dataset.save_to_disk(os.path.join(CACHE_DIR, 'aharley/rvl_cdip'))
else:
    # Load the dataset from the cache
    dataset = load_dataset(os.path.join(CACHE_DIR, 'aharley/rvl_cdip'))

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
        custom_config = '--dpi 300 --psm 3'

        # Use pytesseract to perform OCR on the processed image
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng', config=custom_config)

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
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", cache_dir=BART_CACHE_DIR)
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

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", cache_dir=RESNET_CACHE_DIR)
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", cache_dir=RESNET_CACHE_DIR)

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

def insert_file(file_path):
    # Insert file data into the in-memory database or any other data storage mechanism you prefer
    # ...
    pass


def get_all_files_from_db():
    # Retrieve all file data from the in-memory database or any other data storage mechanism you are using
    # ...
    return []


def get_file_from_db(file_id):
    # Retrieve file data from the in-memory database or any other data storage mechanism based on the file ID
    # ...
    return None

# Main loop
while True:
    print("Launching server, first time it will be slow, since it's building data and model cache don't worry it will be faster next time")
    file_paths_input = input("Enter the file paths (comma-separated), or 'q' to quit: ")

    if file_paths_input.lower() == 'q':
        break

    file_paths = [path.strip() for path in file_paths_input.split(",")]

    cleaned_extracted_text = extract_text(file_paths)

    if cleaned_extracted_text:
        json_output = json.dumps(cleaned_extracted_text, indent=4)
        print("Extracted Text (JSON):")
        print(json_output)
    else:
        print("No text extracted.")