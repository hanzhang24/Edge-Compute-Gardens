# %% [markdown]
# # Garden Recommender System - Training Notebook
# 
# This notebook will help you train the garden recommender model in Google Colab.

# %%
# Install dependencies
!pip install torch torchvision pandas numpy Pillow tqdm requests

# %%
# Create necessary directories
!mkdir -p data/training/images

# %%
# Upload your training data
from google.colab import files
print("Please upload your training_dataset.csv file:")
files.upload()

# %%
# Upload your images (if needed)
print("Please upload your training images to the data/training/images directory:")
files.upload()

# %%
# Import necessary modules
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import math

# %%
# Copy model files to Colab
!wget https://raw.githubusercontent.com/your-repo/garden_project/main/models/garden_model.py
!wget https://raw.githubusercontent.com/your-repo/garden_project/main/models/feature_extractor.py

# %%
# Train the model
from garden_model import GardenRecommender
from feature_extractor import GardenFeatureExtractor

# Initialize model
model = GardenRecommender()

# Create dataset
class GardenDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.feature_extractor = GardenFeatureExtractor()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract features from image using ResNet
        image_features = self.feature_extractor.extract_features(row['image_path'])
        image_features = torch.FloatTensor(image_features)
            
        # Get environmental data (4 features as expected by the model)
        # Calculate day sine/cosine from the date
        date = datetime.strptime(row['date'], "%Y-%m-%d")
        day_of_week = date.weekday()  # 0 is Monday, 6 is Sunday
        day_sin = math.sin(2 * math.pi * day_of_week / 7)
        day_cos = math.cos(2 * math.pi * day_of_week / 7)
        
        env_data = torch.tensor([
            float(row['avg_temp']),      # Temperature
            float(row['avg_humidity']),  # Humidity
            day_sin,                     # Day sine
            day_cos                      # Day cosine
        ], dtype=torch.float32)
        
        # Get target (rating normalized to 0-1)
        target = torch.tensor(float(row['rating']) / 5.0, dtype=torch.float32)
        
        return image_features, env_data, target

# Create data loader
dataset = GardenDataset('training_dataset.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (image_features, env_data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(image_features, env_data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# %%
# Save the trained model
torch.save(model.state_dict(), 'garden_model.pth')

# %%
# Download the trained model
from google.colab import files
files.download('garden_model.pth')

# %% [markdown]
# ## Model Training Complete!
# 
# The trained model has been saved and downloaded to your local machine. 