import torch
import torch.nn as nn
import numpy as np
from feature_extractor import GardenFeatureExtractor
from datetime import datetime
import math

class GardenRecommender(nn.Module):
    def __init__(self, feature_dim=2053):  # 2048 ResNet + 5 custom features
        super(GardenRecommender, self).__init__()
        
        # Dimensions
        self.feature_dim = feature_dim
        self.env_dim = 4  # temp, humidity, day_sin, day_cos
        
        # Image feature processing
        self.image_network = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Environmental data processing
        self.env_network = nn.Sequential(
            nn.Linear(self.env_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_network = nn.Sequential(
            nn.Linear(128 + 16, 64),  # Combining image and env features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Output single score
            nn.Sigmoid()  # Scale to 0-1
        )
        
        self.feature_extractor = GardenFeatureExtractor()

    def _encode_cyclical_feature(self, value, max_value):
        """Encode a cyclical feature using sine and cosine"""
        sin = math.sin(2 * math.pi * value / max_value)
        cos = math.cos(2 * math.pi * value / max_value)
        return sin, cos

    def parse_date(self, date_str):
        """
        Parse date string in format (YYYY-MM-DD)
        Returns the day of week (0-6, where 0 is Monday)
        """
        try:
            # Parse the date string
            date = datetime.strptime(date_str, "%Y-%m-%d")
            return date.weekday()
        except ValueError:
            # If parsing fails, return Wednesday (3) as a fallback
            print(f"Warning: Could not parse date '{date_str}', using default value")
            return 3

    def _process_environmental_data(self, temperature, humidity, date_str=None):
        """Process environmental and temporal data into features"""
        features = []
        
        # Basic environmental features
        features.extend([
            temperature / 100.0,  # Normalize temperature
            humidity / 100.0,     # Normalize humidity
        ])
        
        # Day of week (cyclical)
        if date_str is not None:
            day_of_week = self.parse_date(date_str)
        else:
            day_of_week = 3  # Default to Wednesday if no date provided
        
        day_sin, day_cos = self._encode_cyclical_feature(day_of_week, 7)
        features.extend([day_sin, day_cos])
        
        return torch.FloatTensor(features)

    def forward(self, image_features, env_data):
        # Process image features
        img_out = self.image_network(image_features)
        
        # Process environmental data
        env_out = self.env_network(env_data)
        
        # Combine features
        combined = torch.cat((img_out, env_out), dim=1)
        
        # Final processing
        output = self.combined_network(combined)
        return output
    
    def get_recommendation(self, image_input=None, temperature=None, humidity=None, date_str=None):
        """
        Generate a recommendation given available data. Any parameter can be None.
        Args:
            image_input: Either a path to an image file or image bytes (optional)
            temperature: Temperature in Fahrenheit (optional)
            humidity: Humidity percentage (optional)
            date_str: Date string in format 'YYYY-MM-DD' (optional)
        Returns: (score, recommendation_text)
        """
        # Set model to training mode to enable dropout
        self.train()
        
        # Handle image features
        if image_input is not None:
            image_features = self.feature_extractor.extract_features(image_input)
            image_features = torch.FloatTensor(image_features).unsqueeze(0)
        else:
            # Create a zero tensor with the same shape as image features
            image_features = torch.zeros(1, self.feature_dim)
        
        # Process environmental data
        if temperature is None:
            temperature = 0  # Default value
        if humidity is None:
            humidity = 0  # Default value
            
        env_data = self._process_environmental_data(
            temperature, humidity, date_str
        ).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            score = self.forward(image_features, env_data).item()
        
        # Generate recommendation text
        recommendation = self._generate_recommendation_text(
            score, temperature, humidity, date_str
        )
        
        return score, recommendation
    
    def _generate_recommendation_text(self, score, temperature, humidity, date_str=None):
        """Generate a detailed recommendation based on available data"""
        if score > 0.8:
            base_text = "Perfect day to visit Duke Gardens! 🌸"
        elif score > 0.6:
            base_text = "Good conditions for a garden visit! 🌺"
        elif score > 0.4:
            base_text = "Decent conditions for visiting. 🌿"
        elif score > 0.2:
            base_text = "Maybe wait for better conditions. 🌧"
        else:
            base_text = "Not recommended today. ⛈"
            
        # Build detailed conditions text
        conditions = []
        
        if temperature != 0:  # Only add if temperature was provided
            conditions.append(f"Temperature: {temperature:.1f}°F")
        if humidity != 0:  # Only add if humidity was provided
            conditions.append(f"Humidity: {humidity:.1f}%")
        
        if date_str:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                conditions.append(f"Day: {date.strftime('%A')}")
            except ValueError:
                pass
            
        conditions_text = "\n".join(conditions) if conditions else "Limited data available"
        return f"{base_text}\n\n{conditions_text}"

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the garden recommender model
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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