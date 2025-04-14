import torch
from garden_model import GardenRecommender
from feature_extractor import GardenFeatureExtractor
from PIL import Image
import numpy as np
from datetime import datetime
import math

def predict_garden_rating(image_path, avg_temp, avg_humidity, date):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model = GardenRecommender().to(device)
    model.load_state_dict(torch.load('garden_model_golden.pth'))
    model.eval()  # Set to evaluation mode
    
    # Initialize feature extractor
    feature_extractor = GardenFeatureExtractor()
    
    # Extract image features
    image_features = feature_extractor.extract_features(image_path)
    image_features = torch.FloatTensor(image_features).to(device)
    
    # Prepare environmental data
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    day_of_week = date_obj.weekday()
    day_sin = math.sin(2 * math.pi * day_of_week / 7)
    day_cos = math.cos(2 * math.pi * day_of_week / 7)
    
    env_data = torch.tensor([
        float(avg_temp),
        float(avg_humidity),
        day_sin,
        day_cos
    ], dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_features.unsqueeze(0), env_data.unsqueeze(0))
        # Convert prediction back to 1-5 scale (since we normalized to 0-1 during training)
        rating = prediction.item() * 5.0
    
    return rating

if __name__ == "__main__":
    # Example usage
    image_path = "uploads/image.jpg"  # Replace with your image path
    avg_temp = 32  # Replace with actual temperature
    avg_humidity = 1  # Replace with actual humidity
    date = "2025-03-20"  # Replace with actual date
    
    rating = predict_garden_rating(image_path, avg_temp, avg_humidity, date)
    print(f"Predicted garden rating: {rating:.2f}/5.0") 