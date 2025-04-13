import pandas as pd
import os
import requests
from PIL import Image
import io
import numpy as np
from tqdm import tqdm

def get_target_image_size():
    """Get the target image size from the uploads folder"""
    image_path = os.path.join('uploads', 'image.jpg')
    with Image.open(image_path) as img:
        return img.size

def download_and_normalize_image(url, target_size):
    """Download and normalize an image to the target size"""
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error processing image {url}: {e}")
        return None

def prepare_training_data():
    """Prepare the training data with normalized images and simplified weather features"""
    # Load the combined data
    df = pd.read_csv('data/training/combined_weather_reviews.csv')
    
    # Get target image size
    target_size = get_target_image_size()
    print(f"Target image size: {target_size}")
    
    # Create directories for normalized images
    os.makedirs('data/training/images', exist_ok=True)
    
    # Prepare new dataset
    training_data = []
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Skip rows with missing images
        if pd.isna(row['images']):
            continue
            
        # Get the first image URL from the list
        image_urls = str(row['images']).split(', ')
        if not image_urls or not image_urls[0]:
            continue
            
        # Download and normalize the first image
        img = download_and_normalize_image(image_urls[0], target_size)
        if img is None:
            continue
            
        # Save the normalized image
        image_path = os.path.join('data/training/images', f'image_{idx}.jpg')
        img.save(image_path)
        
        # Create training sample
        sample = {
            'image_path': image_path,
            'rating': row['rating'],
            'avg_temp': row['avg_temp'],
            'avg_humidity': row['avg_humidity'],
            'date': row['date']  # Assuming the date is in the original dataset
        }
        training_data.append(sample)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Save the training dataset
    training_df.to_csv('data/training/training_dataset.csv', index=False)
    print(f"Created training dataset with {len(training_df)} samples")

if __name__ == "__main__":
    prepare_training_data() 