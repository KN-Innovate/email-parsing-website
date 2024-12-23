import mailbox
import pandas as pd
import torch
from pymongo import MongoClient


# Function to parse an MBOX file
from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("ab-ai/PII-Model-Phi3-Mini")
# model = AutoModelForTokenClassification.from_pretrained("ab-ai/PII-Model-Phi3-Mini")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# MongoDB database and collection names
db_name = "kni"
collection_name = "MailData"

# Use a pipeline for token classification
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)



def parse_mbox(file_path):
    """
    Parses the MBOX file and extracts email details.

    Parameters:
        file_path (str): Path to the MBOX file.

    Returns:
        list: A list of dictionaries containing email details.
    """
    try:
        print("Parsing MBOX file...")
        mbox = mailbox.mbox(file_path)
    except Exception as e:
        print(f"Error loading MBOX file: {e}")
        return []

    # Initialize an empty list to store email data
    email_data = []

    for message in mbox:
        try:
            
            # Extract text content
            body = get_email_body(message)
            if not body:
                continue
            
            # Use the NER pipeline to extract PII
            body_filtered = mask_pii(body, aggregate_redaction=False)

            # print("body",body.lower(),body_filtered)

            email_info = {
                "Subject": message["subject"],
                "From": message["from"],
                "To": message["to"],
                "Date": message["date"],
                "Body": body_filtered,
                "Label": message["X-Gmail-Labels"],
            }
            email_data.append(email_info)
            return email_data
            break
        except Exception as e:
            print(f"Error processing message: {e}")

    return email_data



def mask_pii(text, aggregate_redaction=True):
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

    masked_text = list(text)
    is_redacting = False
    redaction_start = 0
    current_pii_type = ''

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:  # Special token
            continue

        label = predictions[0][i].item()
        if label != model.config.label2id['O']:  # Non-O label
            pii_type = model.config.id2label[label]
            if not is_redacting:
                is_redacting = True
                redaction_start = start
                current_pii_type = pii_type
            elif not aggregate_redaction and pii_type != current_pii_type:
                # End current redaction and start a new one
                apply_redaction(masked_text, redaction_start, start, current_pii_type, aggregate_redaction)
                redaction_start = start
                current_pii_type = pii_type
        else:
            if is_redacting:
                apply_redaction(masked_text, redaction_start, end, current_pii_type, aggregate_redaction)
                is_redacting = False

    # Handle case where PII is at the end of the text
    if is_redacting:
        apply_redaction(masked_text, redaction_start, len(masked_text), current_pii_type, aggregate_redaction)

    return ''.join(masked_text)

def apply_redaction(masked_text, start, end, pii_type, aggregate_redaction):
    for j in range(start, end):
        masked_text[j] = ''
    if aggregate_redaction:
        masked_text[start] = '[redacted]'
    else:
        masked_text[start] = f'[{pii_type}]'


# Function to extract email body (text/plain or text/html)
def get_email_body(message):
    """
    Extracts the body content of an email.

    Parameters:
        message (email.message.Message): The email message object.

    Returns:
        str: The email body content as plain text or HTML, if available.
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


# Function to save data to CSV
def save_to_csv(email_data, output_file):
    """
    Saves the parsed email data to a CSV file.

    Parameters:
        email_data (list): List of dictionaries containing email details.
        output_file (str): Path to the output CSV file.
    """
    try:
        df = pd.DataFrame(email_data)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


# Function to connect to MongoDB
def connect_to_mongodb(db_name, collection_name):
    """
    Connects to the MongoDB database and returns the collection object.

    Parameters:
        db_name (str): Name of the database.
        collection_name (str): Name of the collection.

    Returns:
        Collection: MongoDB collection object.
    """
    try:
        # Connect to MongoDB (adjust the connection string if needed)
        client = MongoClient("mongodb+srv://kn:akgAjBRLB9rkc26@cluster0.tugfp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        print("Connected to MongoDB")
        db = client[db_name]
        collection = db[collection_name]
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


# Function to insert email data into MongoDB
def insert_into_mongodb(collection, email_data):
    """
    Inserts email data into a MongoDB collection.

    Parameters:
        collection: MongoDB collection object.
        email_data (list): List of dictionaries containing email details.
    """
    try:
        # Insert multiple documents into the collection
        collection.insert_many(email_data)
        print("Email data successfully inserted into MongoDB")
    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")


# Main script
if __name__ == "__main__":
    print("Running the script...")
    # Specify the device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Replace these paths with your file paths
    mbox_file_path = "data_to_be_sent copy.mbox"  # Path to your MBOX file
    output_csv_path = "emailsdata_neel_sir.csv"  # Desired output CSV file path

    # MongoDB database and collection names
    db_name = "kni"
    collection_name = "MailData"

    # Parse MBOX file
    print("Step 1: Parsing MBOX file...")
    emails = parse_mbox(mbox_file_path)

    # Check if emails were parsed successfully
    if emails:
        print(f"Step 2: {len(emails)} emails parsed successfully!")

        # Save parsed data to a CSV file
        print(f"Step 3: Saving emails to CSV file: {output_csv_path}...")
        # save_to_csv(emails, output_csv_path)

        # Connect to MongoDB
        print("Step 4: Connecting to MongoDB...")
        collection = connect_to_mongodb(db_name, collection_name)

        # Corrected condition to check for collection
        if collection is not None:
            # Insert parsed data into MongoDB
            print("Step 5: Inserting emails into MongoDB...")
            insert_into_mongodb(collection, emails)
        else:
            print("Failed to connect to MongoDB.")
    else:
        print("No emails were parsed.")
