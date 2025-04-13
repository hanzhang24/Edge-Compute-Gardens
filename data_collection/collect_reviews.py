import requests
import pandas as pd
import os
from datetime import datetime
import time
import random
from urllib.request import urlretrieve
import uuid

class TripAdvisorReviewCollector:
    def __init__(self):
        self.base_url = "https://api.content.tripadvisor.com/api/v1"
        self.api_key = "YOUR_TRIPADVISOR_API_KEY"  # You'll need to get this from TripAdvisor
        self.location_id = "55107"  # Duke Gardens location ID
        self.headers = {
            "accept": "application/json"
        }
        
    def fetch_reviews(self):
        """Fetch reviews from TripAdvisor API"""
        try:
            # First get location details
            print("Fetching location details...")
            location_url = f"{self.base_url}/location/{self.location_id}/details"
            response = requests.get(
                location_url,
                headers=self.headers,
                params={"key": self.api_key}
            )
            location_data = response.json()
            
            if 'error' in location_data:
                print(f"Error: {location_data['error']}")
                return pd.DataFrame()
            
            # Then get reviews
            print("Fetching reviews...")
            reviews_url = f"{self.base_url}/location/{self.location_id}/reviews"
            response = requests.get(
                reviews_url,
                headers=self.headers,
                params={
                    "key": self.api_key,
                    "language": "en"
                }
            )
            reviews_data = response.json()
            
            if 'error' in reviews_data:
                print(f"Error: {reviews_data['error']}")
                return pd.DataFrame()
            
            reviews_list = []
            for review in reviews_data['data']:
                # Get photos from the review
                photo_urls = []
                if 'photos' in review:
                    photo_urls = [photo['images']['original']['url'] for photo in review['photos']]
                
                reviews_list.append({
                    'date': review['published_date'],
                    'rating': review['rating'],
                    'review_text': review['text'],
                    'photo_urls': photo_urls
                })
                print(f"Found review from {review['published_date']} with {len(photo_urls)} photos")
            
            return pd.DataFrame(reviews_list)
            
        except Exception as e:
            print(f"Error fetching reviews: {e}")
            return pd.DataFrame()
    
    def download_photos(self, df, output_dir):
        """Download photos from reviews to a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, row in df.iterrows():
            if row['photo_urls']:  # Only process rows with photos
                for photo_url in row['photo_urls']:
                    # Create unique filename using UUID
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"photo_{unique_id}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    try:
                        urlretrieve(photo_url, filepath)
                        print(f"Downloaded {filename}")
                    except Exception as e:
                        print(f"Error downloading {photo_url}: {e}")
                    
                    time.sleep(random.uniform(1, 3))
    
    def save_data(self, df, output_file):
        """Save the collected data to a CSV file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if df.empty:
            print("No data to save!")
            return
        
        try:
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} entries to {output_file}")
            print(f"CSV file created at: {os.path.abspath(output_file)}")
            
            # Print sample of the data
            print("\nSample of saved data:")
            print(df.head())
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempted to save to: {os.path.abspath(output_file)}")

def main():
    try:
        # Initialize collector
        collector = TripAdvisorReviewCollector()
        
        # Fetch reviews
        print("Starting to fetch reviews...")
        df = collector.fetch_reviews()
        
        if not df.empty:
            # Save review data
            output_file = os.path.join('data', 'duke_gardens_reviews.csv')
            collector.save_data(df, output_file)
            
            # Download photos
            print("Downloading photos...")
            collector.download_photos(df, 'data/review_photos')
            
            print("Data collection complete!")
        else:
            print("No reviews or photos found.")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 