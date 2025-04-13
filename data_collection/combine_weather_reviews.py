import pandas as pd
import os
from datetime import datetime
import glob

def load_weather_data():
    """Load and combine all weather data files"""
    weather_files = glob.glob('data/CRND0103-*.txt')
    all_weather_data = []
    
    for file in weather_files:
        # Read the weather data file
        df = pd.read_csv(file, delim_whitespace=True, header=None)
        
        # Extract relevant columns (date, temperature, humidity)
        # Based on the file format, columns are:
        # 0: station_id, 1: date, 2: latitude, 3: longitude, 4: elevation
        # 5: max_temp, 6: min_temp, 7: avg_temp, 8: precipitation
        # 9: solar_radiation, 10: wind_speed, 11: max_humidity, 12: min_humidity
        # 13: avg_humidity, etc.
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df[1], format='%Y%m%d')
        
        # Create daily weather record
        daily_weather = {
            'date': df['date'],
            'max_temp': df[5],
            'min_temp': df[6],
            'avg_temp': df[7],
            'max_humidity': df[11],
            'min_humidity': df[12],
            'avg_humidity': df[13]
        }
        
        all_weather_data.append(pd.DataFrame(daily_weather))
    
    # Combine all weather data
    weather_df = pd.concat(all_weather_data, ignore_index=True)
    return weather_df

def load_reviews():
    """Load and process review data"""
    reviews_df = pd.read_csv('data/duke_gardens_google_reviews.csv')
    
    # Convert date to datetime
    reviews_df['date'] = pd.to_datetime(reviews_df['date'])
    
    # Extract just the date part (remove time)
    reviews_df['date'] = reviews_df['date'].dt.date
    
    return reviews_df

def combine_data(weather_df, reviews_df):
    """Combine weather and review data"""
    # Convert weather date to date type for matching
    weather_df['date'] = weather_df['date'].dt.date
    
    # Merge on date
    combined_df = pd.merge(reviews_df, weather_df, on='date', how='inner')
    
    # Create final dataset structure
    final_df = combined_df[[
        'date',
        'rating',
        'avg_temp',
        'avg_humidity',
        'max_temp',
        'min_temp',
        'max_humidity',
        'min_humidity',
        'images'
    ]]
    
    return final_df

def main():
    # Load data
    weather_df = load_weather_data()
    reviews_df = load_reviews()
    
    # Combine data
    combined_df = combine_data(weather_df, reviews_df)
    
    # Save combined dataset
    os.makedirs('data/training', exist_ok=True)
    combined_df.to_csv('data/training/combined_weather_reviews.csv', index=False)
    print(f"Created training dataset with {len(combined_df)} samples")

if __name__ == "__main__":
    main() 