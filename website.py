from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from datetime import datetime
import shutil
import torch
from garden_model import GardenRecommender

app = Flask(__name__)
LATENCY_LOG = "latency_log.txt"
UPLOAD_FOLDER = "uploads"
DEFAULT_IMAGE = "image.jpg"
MODEL_PATH = "garden_model.pth"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize the model
model = GardenRecommender()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Store the latest data
latest_data = {
    "image": DEFAULT_IMAGE,
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
if not os.path.exists(os.path.join(UPLOAD_FOLDER, DEFAULT_IMAGE)):
    # Create an empty image file if it doesn't exist
    with open(os.path.join(UPLOAD_FOLDER, DEFAULT_IMAGE), 'w') as f:
        pass

@app.route('/')
def index():
    current_image = latest_data["image"] if os.path.exists(os.path.join(UPLOAD_FOLDER, latest_data["image"])) else DEFAULT_IMAGE
    return render_template('index.html', 
                         latest_image=current_image,
                         recommendation=latest_data["recommendation"],
                         data=latest_data["data"])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def update_recommendation():
    """Update the recommendation if we have all necessary data"""
    if latest_data["temperature"] is not None and latest_data["humidity"] is not None:
        image_path = os.path.join(UPLOAD_FOLDER, DEFAULT_IMAGE)
        if os.path.exists(image_path):
            score, recommendation = model.get_recommendation(
                image_path,
                latest_data["temperature"],
                latest_data["humidity"]
            )
            latest_data["recommendation"] = recommendation
            
            # Update the data table
            latest_data["data"]["rows"] = [
                ["Temperature", f"{latest_data['temperature']:.1f}°F"],
                ["Humidity", f"{latest_data['humidity']:.1f}%"],
                ["Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            ]

@app.route('/upload', methods=['POST'])
def receive_data():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Save as image.jpg (overwriting the existing one)
            file_path = os.path.join(UPLOAD_FOLDER, DEFAULT_IMAGE)
            file.save(file_path)
            
            # Also save a timestamped version for history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_filename = f"{timestamp}_{file.filename}"
            history_path = os.path.join(UPLOAD_FOLDER, history_filename)
            shutil.copy2(file_path, history_path)
            
            # Update latest image
            latest_data["image"] = DEFAULT_IMAGE
            
            # Update recommendation with new image
            update_recommendation()
            
            return jsonify({"status": "success", "filepath": file_path})
        else:
            return jsonify({"error": "Invalid file type"}), 400
            
    elif request.is_json:
        data = request.get_json()
        
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
