
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
from werkzeug.utils import secure_filename
import uuid  # For generating unique IDs
import mailbox
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Database configuration
db_name = "kni"
collection_name = "MailData"

# Global variables to store parsed emails and PII count
parsed_emails = []
pii_count = 0

# Initialize PII model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

def connect_to_mongodb(db_name, collection_name):
    """
    Connects to the MongoDB database and returns the collection object.
    Also ensures necessary indexes are created.
    """
    try:
        # MongoDB URI should be stored securely, e.g., as an environment variable
        # For demonstration, it's hardcoded here (Not recommended for production)
        mongo_uri = "mongodb+srv://kn:akgAjBRLB9rkc26@cluster0.tugfp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set.")
        
        client = MongoClient(mongo_uri)
        print("Connected to MongoDB")
        db = client[db_name]
        collection = db[collection_name]

        # Create indexes if they don't exist
        collection.create_index("hotel_name")
        collection.create_index("From")
        collection.create_index("Date")

        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def mask_pii(text, aggregate_redaction=True):
    """
    Masks PII in the given text and returns the masked text along with the count of PII redactions.
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert token predictions to word predictions
    encoded_inputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = encoded_inputs['offset_mapping']

    pii_spans = []
    current_pii = None
    current_pii_type = ''

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue  # Special token

        label = predictions[0][i].item()
        if label != model.config.label2id.get('O', 0):  # Non-O label
            pii_type = model.config.id2label.get(label, 'PII')
            if not current_pii:
                current_pii = [start, end]
                current_pii_type = pii_type
            else:
                # Check if the current PII type matches
                if not aggregate_redaction and pii_type != current_pii_type:
                    pii_spans.append(tuple(current_pii))
                    current_pii = [start, end]
                    current_pii_type = pii_type
                else:
                    # Extend the current PII span
                    current_pii[1] = end
        else:
            if current_pii:
                pii_spans.append(tuple(current_pii))
                current_pii = None

    # Handle case where PII is at the end of the text
    if current_pii:
        pii_spans.append(tuple(current_pii))

    # Apply redaction
    masked_text = text
    for start, end in reversed(pii_spans):
        masked_text = masked_text[:start] + '[redacted]' + masked_text[end:]

    pii_count = len(pii_spans)
    return masked_text, pii_count

def get_email_body(message):
    """
    Extracts the body content of an email.
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

def parse_mbox(file_path, hotel_name):
    """
    Parses the MBOX file and extracts email details along with PII redaction count.
    Associates each email with a specific hotel using hotel_name.
    """
    try:
        print("Parsing MBOX file...")
        mbox = mailbox.mbox(file_path)
    except Exception as e:
        print(f"Error loading MBOX file: {e}")
        return [], 0

    email_data = []
    total_pii_count = 0

    for message in mbox:
        try:
            body = get_email_body(message)
            if not body:
                continue

            # Mask PII in the email body
            body_filtered, pii_in_email = mask_pii(body, aggregate_redaction=False)
            total_pii_count += pii_in_email

            email_info = {
                "id": str(uuid.uuid4()),  # Assign a unique ID
                "hotel_name": hotel_name, # Associate with hotel
                "Label": message["X-Gmail-Labels"],
                "From": message["from"],
                "To": message["to"],      # Ensure 'To' field is captured
                "Date": message["date"],
                "Subject": message["subject"],
                "Body": body_filtered,
            }
            email_data.append(email_info)
        except Exception as e:
            print(f"Error processing message: {e}")

    return email_data, total_pii_count  # Return after processing all emails

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the parsed emails page
@app.route('/parsed-emails')
def parsed_emails_page():
    hotel_name = request.args.get('hotel_name')
    if not hotel_name:
        return "Hotel name not provided.", 400

    # Filter emails based on hotel_name
    filtered_emails = [email for email in parsed_emails if email['hotel_name'] == hotel_name]
    total = len(filtered_emails)

    # Calculate PII count for the filtered emails
    pii_count = sum(email['Body'].count('[redacted]') for email in filtered_emails)

    return render_template('parsed-emails.html', 
                           emails=filtered_emails, 
                           pii_count=pii_count, 
                           hotel_name=hotel_name)

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    global parsed_emails
    global pii_count
    try:
        file = request.files['file']
        hotel_name = request.form.get('hotel_name')
        if not hotel_name:
            return jsonify({"success": False, "message": "Hotel name is required."}), 400
        
        if file:
            # Sanitize hotel_name
            hotel_name = hotel_name.strip()
            if not hotel_name:
                return jsonify({"success": False, "message": "Hotel name cannot be empty."}), 400

            # Check for duplicate hotel_name in parsed_emails
            existing_hotel = any(email['hotel_name'] == hotel_name for email in parsed_emails)
            if existing_hotel:
                # Append a unique identifier to hotel_name
                unique_identifier = str(uuid.uuid4())[:8]
                hotel_name = f"{hotel_name}_{unique_identifier}"
                print(f"Duplicate hotel name detected. Renamed to {hotel_name}")

            # Save the uploaded file to a temporary location
            filename = secure_filename(file.filename)
            filepath = os.path.join('/tmp', filename)
            file.save(filepath)
            
            # Parse the MBOX file and mask PII
            parsed_emails, pii_count = parse_mbox(filepath, hotel_name)

            print(f"Parsed {len(parsed_emails)} emails with {pii_count} PII redactions.")

            # Return JSON response instead of redirect
            return jsonify({"success": True, "hotel_name": hotel_name})
        else:
            return jsonify({"success": False, "message": "No file uploaded."}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Route to save parsed emails to MongoDB
@app.route('/save-emails', methods=['POST'])
def save_emails():
    global parsed_emails
    global pii_count
    try:
        collection = connect_to_mongodb(db_name, collection_name)
        if collection is not None:
            if parsed_emails:
                # Insert parsed emails into MongoDB
                collection.insert_many(parsed_emails)
                parsed_emails = []  # Clear temporary storage after saving
                pii_count = 0
                return jsonify({"success": True, "message": "Emails saved to database."})
            else:
                return jsonify({"success": False, "message": "No emails to save."})
        else:
            return jsonify({"success": False, "message": "Database connection failed."}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Route to edit an email in the parsed_emails list
@app.route('/edit-email', methods=['POST'])
def edit_email():
    global parsed_emails
    try:
        # Get form data
        email_id = request.form['email_id']
        updated_data = {
            "Label": request.form.get('tags', ''),
            "From": request.form.get('from', ''),
            "To": request.form.get('to', ''),
            "Date": request.form.get('date', ''),
            "Subject": request.form.get('subject', ''),
            "Body": request.form.get('body', ''),
        }

        # Find the email by ID and update it
        for email in parsed_emails:
            if email['id'] == email_id:
                email.update(updated_data)
                return jsonify({"success": True, "message": "Email updated successfully."})

        return jsonify({"success": False, "message": "Email not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Route to delete an email from the parsed_emails list
@app.route('/delete-email', methods=['POST'])
def delete_email():
    global parsed_emails
    try:
        data = request.get_json()
        email_id = data['email_id']
        # Find the email by ID and remove it
        for index, email in enumerate(parsed_emails):
            if email['id'] == email_id:
                del parsed_emails[index]
                return jsonify({"success": True, "message": "Email deleted successfully."})
        return jsonify({"success": False, "message": "Email not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
