import requests
import pandas as pd
from datetime import datetime, timedelta
import os

class NCEIDataCollector:
    def __init__(self):
        self.base_url = "https://www.ncei.noaa.gov/access/crn/qcdatasets"
        self.station_id = "1347"  # NC Durham 11 W (Duke Forest)
        
    def fetch_daily_data(self, start_date, end_date):
        """
        Fetch daily temperature and humidity data from NCEI
        Args:
            start_date: datetime object
            end_date: datetime object
        Returns:
            DataFrame with columns: date, temperature, humidity
        """
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                # Format URL parameters
                year = current_date.year
                month = current_date.month
                
                # NCEI data is organized by year/month
                url = f"{self.base_url}/hourly/{year:04d}/{self.station_id}.{year:04d}{month:02d}.hourly.txt"
                print(f"Fetching from: {url}")
                
                response = requests.get(url)
                if response.status_code == 200:
                    # Process the data
                    lines = response.text.split('\n')
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            fields = line.split()
                            if len(fields) >= 8:
                                # NCEI format: WBANNO,LST_DATE,CRX_VN,LST_TIME,T_CALC,T_HR_AVG,T_MAX,T_MIN,P_CALC,SOLARAD,SOLARAD_FLAG,SOLARAD_MAX,SOLARAD_MAX_FLAG,SOLARAD_MIN,SOLARAD_MIN_FLAG,SUR_TEMP_TYPE,SUR_TEMP,SUR_TEMP_FLAG,SUR_TEMP_MAX,SUR_TEMP_MAX_FLAG,SUR_TEMP_MIN,SUR_TEMP_MIN_FLAG,RH_HR_AVG,RH_HR_AVG_FLAG,SOIL_MOISTURE_5,SOIL_MOISTURE_10,SOIL_MOISTURE_20,SOIL_MOISTURE_50,SOIL_MOISTURE_100,SOIL_TEMP_5,SOIL_TEMP_10,SOIL_TEMP_20,SOIL_TEMP_50,SOIL_TEMP_100
                                date_str = fields[1]  # LST_DATE
                                time_str = fields[3]  # LST_TIME
                                temp = float(fields[4])  # T_CALC (Celsius)
                                humidity = float(fields[22])  # RH_HR_AVG
                                
                                # Convert date and time
                                timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str}:00"
                                
                                data.append({
                                    'date': timestamp,
                                    'temperature': (temp * 9/5) + 32,  # Convert to Fahrenheit
                                    'humidity': humidity
                                })
                                print(f"Collected data for {timestamp}")
                
            except Exception as e:
                print(f"Error fetching data for {current_date}: {e}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        return pd.DataFrame(data)

    def save_data(self, df, output_file):
        """Save the collected data to a CSV file"""
        os.makedirs('data', exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")

def main():
    # Initialize collector
    collector = NCEIDataCollector()
    
    # Set date range (e.g., last month for testing)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Start with a month of data for testing
    
    # Fetch data
    print(f"Fetching data from {start_date} to {end_date}...")
    df = collector.fetch_daily_data(start_date, end_date)
    
    # Save data
    collector.save_data(df, 'data/duke_forest_weather.csv')
    print("Data collection complete!")

if __name__ == "__main__":
    main() 