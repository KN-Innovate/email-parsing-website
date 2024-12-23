from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS  # type: ignore
from pymongo import MongoClient
from bson.objectid import ObjectId
import os

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# MongoDB connection details
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client['email_repository']  # Replace with your database name
collection = db['emails']  # Replace with your collection name

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the parsed emails page
@app.route('/parsed-emails')
def parsed_emails():
    # Fetch data from MongoDB to display in the table
    emails = list(collection.find({}, {'_id': 1, 'tags': 1, 'from': 1, 'date': 1, 'subject': 1, 'body': 1}))
    for email in emails:
        email['_id'] = str(email['_id'])  # Convert ObjectId to string for rendering
    return render_template('parsed-emails.html', emails=emails)

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if file:
            # Perform necessary processing (e.g., parse MBOX file here)
            # Redirect to parsed-emails page after successful upload
            return redirect(url_for('parsed_emails'))
        else:
            return jsonify({"success": False, "message": "No file uploaded."}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Route to save an email to the database
@app.route('/save-email', methods=['POST'])
def save_email():
    try:
        # Parse the form data
        email_data = {
            "tags": request.form['tags'],
            "from": request.form['from'],
            "date": request.form['date'],
            "subject": request.form['subject'],
            "body": request.form['body']
        }
        # Insert into MongoDB
        collection.insert_one(email_data)
        return jsonify({"success": True, "message": "Email saved to database."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Route to edit an email in the database
@app.route('/edit-email', methods=['POST'])
def edit_email():
    try:
        email_id = request.form.get('email_id')
        updated_data = {
            "tags": request.form['tags'],
            "from": request.form['from'],
            "date": request.form['date'],
            "subject": request.form['subject'],
            "body": request.form['body']
        }
        # Update the database
        collection.update_one({"_id": ObjectId(email_id)}, {"$set": updated_data})
        return jsonify({"success": True, "message": "Email updated successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Route to delete an email from the database
@app.route('/delete-email', methods=['POST'])
def delete_email():
    try:
        email_id = request.form.get('email_id')
        collection.delete_one({"_id": ObjectId(email_id)})
        return jsonify({"success": True, "message": "Email deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
