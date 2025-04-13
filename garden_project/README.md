# Garden Recommender System

This project implements a machine learning model that recommends garden visit times based on image features and environmental conditions.

## Project Structure

```
garden_project/
├── data/
│   └── training/
│       ├── images/          # Training images
│       └── training_dataset.csv  # Training data
├── models/
│   ├── garden_model.py     # Model architecture
│   └── feature_extractor.py # Feature extraction utilities
├── utils/
│   └── prepare_training_data.py  # Data preparation scripts
├── train_model.py          # Training script
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare training data:
```bash
python utils/prepare_training_data.py
```

3. Train the model:
```bash
python train_model.py
```

## Google Colab Setup

1. Upload the entire `garden_project` folder to your Google Drive
2. Open Google Colab and mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Change to the project directory:
```python
%cd /content/drive/MyDrive/garden_project
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

5. Run the training script:
```python
!python train_model.py
```

## Model Architecture

The model combines:
- Image features extracted using ResNet
- Environmental data (temperature, humidity)
- Temporal features (day of week encoded as sine/cosine)

## Data Format

Training data should be in CSV format with the following columns:
- image_path: Path to the image file
- rating: Rating (1-5)
- avg_temp: Average temperature
- avg_humidity: Average humidity
- date: Date in YYYY-MM-DD format 