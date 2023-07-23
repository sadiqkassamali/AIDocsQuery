import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import textract
import json
import traceback
import os
import re
import PyPDF2
import csv
import pandas as pd
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image_path):
    with Image.open(image_path) as image:
        # Convert to grayscale
        grayscale_image = image.convert("L")

        # Enhance image contrast
        enhanced_image = ImageEnhance.Contrast(grayscale_image).enhance(2.0)

        # Apply image filters
        filtered_image = enhanced_image.filter(ImageFilter.SHARPEN)

        return filtered_image


def extract_text_from_image(file_path):
    try:
        # Preprocess the image
        processed_image = preprocess_image(file_path)

        # Use pytesseract to perform OCR on the processed image
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng')

        return extracted_text.strip()
    except IOError:
        print("Unable to open image file:", file_path)
    except Exception as e:
        traceback.print_exc()


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


def extract_text_from_doc(file_path):
    try:
        # Use textract to extract text from DOC
        extracted_text = textract.process(file_path).decode('utf-8')

        return extracted_text.strip()
    except IOError:
        print("Unable to open DOC file:", file_path)
    except Exception as e:
        traceback.print_exc()


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
    except Exception as e:
        traceback.print_exc()


def extract_text_from_xls(file_path):
    try:
        extracted_text = ""
        data_frame = pd.read_excel(file_path)
        for column in data_frame.columns:
            extracted_text += ' '.join([str(cell) for cell in data_frame[column]]) + '\n'

        return extracted_text.strip()
    except IOError:
        print("Unable to open XLS file:", file_path)
    except Exception as e:
        traceback.print_exc()


def post_process_text(text):
    # Remove non-alphanumeric characters and extra whitespaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()


def extract_text(file_paths):
    extracted_text = {}

    for file_path in file_paths:
        file_metadata = {}

        # Get file information
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_modified = os.path.getmtime(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()

        file_metadata["name"] = file_name
        file_metadata["size"] = file_size
        file_metadata["last_modified"] = datetime.fromtimestamp(file_modified).isoformat()

        if file_extension in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'):
            extracted_text[file_path] = {"image_text": post_process_text(extract_text_from_image(file_path))}
        elif file_extension == '.pdf':
            extracted_text[file_path] = {"pdf_text": post_process_text(extract_text_from_pdf(file_path))}
        elif file_extension in ('.doc', '.docx'):
            extracted_text[file_path] = {"doc_text": post_process_text(extract_text_from_doc(file_path))}
        elif file_extension == '.csv':
            extracted_text[file_path] = {"csv_text": post_process_text(extract_text_from_csv(file_path))}
        elif file_extension in ('.xls', '.xlsx'):
            extracted_text[file_path] = {"xls_text": post_process_text(extract_text_from_xls(file_path))}
        else:
            print("Unsupported file type:", file_extension)

        extracted_text[file_path]["metadata"] = file_metadata

    return extracted_text


# Main loop
while True:
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
