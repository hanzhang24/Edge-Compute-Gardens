from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from gradio_client import Client, handle_file
from datetime import datetime
import shutil
import requests

app = Flask(__name__)
LATENCY_LOG = "latency_log.txt"
UPLOAD_FOLDER = "uploads"
DEFAULT_IMAGE = "image.jpg"
DEFAULT_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, DEFAULT_IMAGE)
GRADIO_API_URL = "Ilovexmlparsing/DukeGardens"  # Replace with your Gradio Space URL

# Initialize Gradio client
client = Client(GRADIO_API_URL)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Store the latest data
latest_data = {
    "image": DEFAULT_IMAGE_PATH,  # Store full path
    "recommendation": None,
    "data": {
        "headers": ["Metric", "Value"],
        "rows": [
            ["Temperature", "N/A"],
            ["Humidity", "N/A"],
            ["Last Updated", "N/A"]
        ]
    },
    "temperature": None,
    "humidity": None
}

# Ensure default image exists in uploads folder
if not os.path.exists(DEFAULT_IMAGE_PATH):
    # Create an empty image file if it doesn't exist
    with open(DEFAULT_IMAGE_PATH, 'w') as f:
        pass

@app.route('/')
def index():
    current_image = latest_data["image"] if os.path.exists(latest_data["image"]) else DEFAULT_IMAGE_PATH
    return render_template('index.html', 
                         latest_image=os.path.basename(current_image),  # Send just filename to template
                         recommendation=latest_data["recommendation"],
                         data=latest_data["data"])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def update_recommendation():
    """Update the recommendation by calling the Gradio API"""
    if client is None:
        latest_data["recommendation"] = "Warning: Gradio Space is not available. Running in offline mode."
        return
        
    try:
        # Prepare image path if it exists
        image_path = latest_data["image"] if os.path.exists(latest_data["image"]) else None
        
        # Call the Gradio API
        result = client.predict(
            image=handle_file(image_path) if image_path else None,
            temperature=latest_data["temperature"] if latest_data["temperature"] is not None else 0,
            humidity=latest_data["humidity"] if latest_data["humidity"] is not None else 0,
            date_str=datetime.now().strftime("%Y-%m-%d"),
            api_name="/predict"
        )
        
        if result:
            latest_data["recommendation"] = result["recommendation"]
            
            # Update the data table
            latest_data["data"]["rows"] = [
                ["Temperature", f"{latest_data['temperature']:.1f}°C" if latest_data['temperature'] is not None else "N/A"],
                ["Humidity", f"{latest_data['humidity']:.1f}%" if latest_data['humidity'] is not None else "N/A"],
                ["Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]
        else:
            latest_data["recommendation"] = "Error: Invalid response from model"
            
    except Exception as e:
        latest_data["recommendation"] = f"Error: {str(e)}"

@app.route('/upload', methods=['POST'])
def receive_data():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Save as image.jpg (overwriting the existing one)
            file.save(DEFAULT_IMAGE_PATH)
            
            # Also save a timestamped version for history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_filename = f"{timestamp}_{file.filename}"
            history_path = os.path.join(UPLOAD_FOLDER, history_filename)
            shutil.copy2(DEFAULT_IMAGE_PATH, history_path)
            
            # Update latest image
            latest_data["image"] = DEFAULT_IMAGE_PATH
            
            # Update recommendation with new image
            update_recommendation()
            
            return jsonify({"status": "success", "filepath": DEFAULT_IMAGE_PATH})
        else:
            return jsonify({"error": "Invalid file type"}), 400
            
    elif request.is_json:
        data = request.get_json()
        print(data)
        # Handle environmental data
        if 'temperature' in data:
            latest_data["temperature"] = float(data['temperature'])
        if 'humidity' in data:
            latest_data["humidity"] = float(data['humidity'])
            
        # Update recommendation if we received new environmental data
        if 'temperature' in data or 'humidity' in data:
            update_recommendation()
            
        # Handle latency data
        if 'latency' in data:
            number = data['latency']
            with open(LATENCY_LOG, "a") as f:
                f.write(f"{number:.6f}\n")
                f.flush()
            print(f"Received latency: {number:.6f}")
            
        return jsonify({"status": "success", "received_data": data})

    return jsonify({"error": "Invalid request format"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
