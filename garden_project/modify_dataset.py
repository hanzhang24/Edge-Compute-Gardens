import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import os

def calculate_image_brightness(image_path):
    """Calculate the average brightness of an image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return 0
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate mean brightness
        brightness = np.mean(gray)
        return brightness
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 0

def modify_dataset(csv_path, output_path):
    # Read the dataset
    df = pd.read_csv(csv_path)
    
    # Define thresholds
    TEMP_THRESHOLD = 10  # °C
    HUMIDITY_THRESHOLD = 15
    BRIGHTNESS_THRESHOLD = 100  # out of 255
    
    # Create a copy of the ratings
    modified_ratings = df['rating'].copy()
    
    # Adjust ratings based on temperature
    low_temp_mask = df['avg_temp'] < TEMP_THRESHOLD
    modified_ratings[low_temp_mask] = modified_ratings[low_temp_mask] - 1
    
    # Adjust ratings based on humidity
    high_humidity_mask = df['avg_humidity'] > HUMIDITY_THRESHOLD
    modified_ratings[high_humidity_mask] = modified_ratings[high_humidity_mask] - 1
    
    # Adjust ratings based on image brightness
    for idx, row in df.iterrows():
        image_path = row['image_path']
        # Make sure the path is correct
        if not os.path.exists(image_path):
            # Try to find the image in the data/training/images directory
            image_name = Path(image_path).name
            image_path = os.path.join('data', 'training', 'images', image_name)
        
        brightness = calculate_image_brightness(image_path)
        if brightness < BRIGHTNESS_THRESHOLD:
            modified_ratings[idx] = modified_ratings[idx] - 1
    
    # Ensure ratings don't go below 1
    modified_ratings = modified_ratings.clip(lower=1.0)
    
    # Create new dataframe with modified ratings
    modified_df = df.copy()
    modified_df['rating'] = modified_ratings
    
    # Save the modified dataset
    modified_df.to_csv(output_path, index=False)
    print(f"Modified dataset saved to {output_path}")
    
    # Print statistics
    print("\nModification Statistics:")
    print(f"Number of images with low temperature (<{TEMP_THRESHOLD}°C): {low_temp_mask.sum()}")
    print(f"Number of images with high humidity (>{HUMIDITY_THRESHOLD}): {high_humidity_mask.sum()}")
    print(f"Original rating range: {df['rating'].min()} - {df['rating'].max()}")
    print(f"Modified rating range: {modified_ratings.min()} - {modified_ratings.max()}")

if __name__ == "__main__":
    input_csv = "data/training/training_dataset.csv"
    output_csv = "data/training/modified_training_dataset.csv"
    modify_dataset(input_csv, output_csv) 