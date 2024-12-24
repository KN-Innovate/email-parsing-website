from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import uuid  # For generating unique IDs
import mailbox
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from werkzeug.utils import secure_filename 
from multiprocessing import set_start_method
from email.header import decode_header, make_header
from bs4 import BeautifulSoup
import re


def decode_mime_string(value):
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except:
        return value

def strip_html(html_text):
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')
    text = text.replace('&nbsp;', ' ')
    return text.strip()

# NEW IMPORTS FOR CLEANING HTML
from bs4 import BeautifulSoup
import re

# Set multiprocessing start method to avoid semaphore warnings
set_start_method('spawn', force=True)

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Database configuration
db_name = "kni"
collection_name = "MailData"

# Global variables
parsed_emails = []
pii_count = 0

# Initialize PII model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# -------------------------------------------------------------------
# 1. CONNECT TO MONGODB
# -------------------------------------------------------------------
def connect_to_mongodb(db_name, collection_name):
    """
    Connects to MongoDB and returns the collection.
    """
    try:
        mongo_uri = "mongodb+srv://kn:akgAjBRLB9rkc26@cluster0.tugfp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(mongo_uri)
        print("Connected to MongoDB")
        db = client[db_name]
        collection = db[collection_name]
        collection.create_index("client_name", background=True)
        collection.create_index("From", background=True)
        collection.create_index("Date", background=True)
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# -------------------------------------------------------------------
# 2. CLEAN / STRIP HTML FUNCTION
# -------------------------------------------------------------------
def strip_html(html_text):
    """
    Removes HTML tags and normalizes whitespace from an HTML string.
    """
    if not html_text:
        return ""
    # Parse the HTML
    soup = BeautifulSoup(html_text, "html.parser")
    # Extract text with spaces separating elements
    text = soup.get_text(separator=" ")
    # Remove &nbsp; or other HTML entities and collapse extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')
    text = text.replace('&nbsp;', ' ')
    return text.strip()

# -------------------------------------------------------------------
# 3. MASK PII
# -------------------------------------------------------------------
def mask_pii(text, aggregate_redaction=True):
    """
    Masks PII in the given text and returns the masked text along with the count of PII redactions.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)

    # Ensure offset mapping matches the truncated input
    encoded_inputs = tokenizer.encode_plus(
        text, 
        return_offsets_mapping=True, 
        add_special_tokens=True, 
        truncation=True, 
        max_length=512
    )
    offset_mapping = encoded_inputs['offset_mapping'][:len(predictions[0])]

    pii_spans = []
    current_pii = None
    current_pii_type = ''

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue

        label = predictions[0][i].item()
        if label != model.config.label2id.get('O', 0):  # Non-O label
            pii_type = model.config.id2label.get(label, 'PII')
            if not current_pii:
                current_pii = [start, end]
                current_pii_type = pii_type
            else:
                if not aggregate_redaction and pii_type != current_pii_type:
                    pii_spans.append(tuple(current_pii))
                    current_pii = [start, end]
                    current_pii_type = pii_type
                else:
                    current_pii[1] = end
        else:
            if current_pii:
                pii_spans.append(tuple(current_pii))
                current_pii = None

    if current_pii:
        pii_spans.append(tuple(current_pii))

    # Apply redaction
    masked_text = text
    for start, end in reversed(pii_spans):
        masked_text = masked_text[:start] + '[redacted]' + masked_text[end:]

    return masked_text, len(pii_spans)

# -------------------------------------------------------------------
# 4. GET EMAIL BODY
# -------------------------------------------------------------------
def get_email_body(message):
    """
    Extracts the body content of an email as text (plain or HTML).
    """
    try:
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    return part.get_payload(decode=True).decode(errors="ignore")
                elif content_type == "text/html":
                    return part.get_payload(decode=True).decode(errors="ignore")
        else:
            return message.get_payload(decode=True).decode(errors="ignore")
    except Exception as e:
        print(f"Error extracting email body: {e}")
        return None

# In-memory registry of uploads
file_registry = {}  # e.g. { client_name: "/tmp/filename" }

# -------------------------------------------------------------------
# 5. UPLOAD ROUTE
# -------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Saves the uploaded file to a temporary location and returns JSON with client_name.
    The actual parsing is handled in /progress via SSE.
    """
    try:
        file = request.files.get('file')
        client_name = request.form.get('client_name')

        if not client_name:
            return jsonify({"success": False, "message": "Client name is required."}), 400
        if not file:
            return jsonify({"success": False, "message": "No file uploaded."}), 400

        # Normalize client_name
        client_name = client_name.strip()
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)

        # Track the file path by client_name
        file_registry[client_name] = filepath

        return jsonify({"success": True, "client_name": client_name})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# -------------------------------------------------------------------
# 6. PROGRESS ROUTE (SSE)
# -------------------------------------------------------------------
@app.route('/progress', methods=['GET'])
def progress():
    """
    Streams parsing progress in real-time using SSE (Server-Sent Events).
    """
    def generate_progress(client_name):
        global parsed_emails, pii_count
        parsed_emails = []
        pii_count = 0

        filepath = file_registry.get(client_name)
        if not filepath:
            yield "data: error - No file found for this client.\n\n"
            return

        # Parse the MBOX file
        try:
            mbox = mailbox.mbox(filepath)
        except Exception as e:
            yield f"data: error - Could not load MBOX: {str(e)}\n\n"
            return

        total_emails = len(mbox)
        if total_emails == 0:
            yield "data: error - MBOX file is empty.\n\n"
            return

        for index, message in enumerate(mbox, start=1):
            try:
                # 1. Decode each header
                subject_decoded = decode_mime_string(message.get("subject"))
                from_decoded = decode_mime_string(message.get("from"))
                date_decoded = decode_mime_string(message.get("date"))
                label_decoded = decode_mime_string(message.get("X-Gmail-Labels"))

                # 2. Extract and clean Body
                raw_body = get_email_body(message)
                if not raw_body:
                    continue
                clean_body = strip_html(raw_body)  # remove HTML tags, etc.

                # 3. Mask PII
                body_filtered, pii_in_email = mask_pii(clean_body, aggregate_redaction=False)
                pii_count += pii_in_email

                # 4. Build email_info dictionary
                email_info = {
                    "id": str(uuid.uuid4()),
                    "client_name": client_name,
                    "Label": label_decoded,
                    "From": from_decoded,
                    "Date": date_decoded,
                    "Subject": subject_decoded,
                    "Body": body_filtered,
                }
                parsed_emails.append(email_info)

            except Exception as exc:
                yield f"data: error - {str(exc)}\n\n"

            # Send progress update
            yield f"data: {index}/{total_emails}\n\n"

        yield "data: completed\n\n"

    client_name = request.args.get('client_name')
    if not client_name:
        return "Client name is required.", 400

    return Response(generate_progress(client_name), mimetype='text/event-stream')

# -------------------------------------------------------------------
# 7. VIEW PARSED EMAILS
# -------------------------------------------------------------------
@app.route('/parsed-emails')
def parsed_emails_page():
    client_name = request.args.get('client_name')
    if not client_name:
        return "Client name not provided.", 400

    filtered = [e for e in parsed_emails if e['client_name'] == client_name]
    count_redacted = sum(e['Body'].count('[redacted]') for e in filtered)

    return render_template(
        'parsed-emails.html',
        emails=filtered,
        pii_count=count_redacted,
        client_name=client_name
    )

# -------------------------------------------------------------------
# 8. SAVE TO DATABASE (OPTIONAL)
# -------------------------------------------------------------------
@app.route('/save-emails', methods=['POST'])
def save_emails():
    global parsed_emails, pii_count
    try:
        coll = connect_to_mongodb(db_name, collection_name)
        if coll is not None and parsed_emails:
            coll.insert_many(parsed_emails)
            parsed_emails.clear()
            pii_count = 0
            return jsonify({"success": True, "message": "Emails saved to database."})
        return jsonify({"success": False, "message": "No emails to save."}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# -------------------------------------------------------------------
# 9. INDEX PAGE
# -------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------------------------------------------------
# 10. RUN THE APP
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8000)
