import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from garden_model import GardenRecommender
from feature_extractor import GardenFeatureExtractor
from tqdm import tqdm
from datetime import datetime
import math

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

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = GardenDataset('data/training/training_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  # Set num_workers to 0 for debugging
    
    # Initialize model
    model = GardenRecommender().to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()  # Using MSE since we're predicting a continuous score
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for image_features, env_data, targets in pbar:
            # Move data to device
            image_features = image_features.to(device)
            env_data = env_data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(image_features, env_data)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'garden_model.pth')
    print("Model saved to garden_model.pth")

if __name__ == "__main__":
    train_model() 