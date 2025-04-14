import gradio as gr
import torch
from garden_model import GardenRecommender
from datetime import datetime
import os
from PIL import Image
import io

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GardenRecommender().to(device)
model.load_state_dict(torch.load('garden_model_golden.pth'))
model.eval()

# Create a directory for temporary images
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

def predict(image, temperature, humidity, date_str=None):
    try:
        # Get current date if not provided
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Handle image
        image_path = None
        if image is not None:
            # Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(TEMP_IMAGE_DIR, f"temp_{timestamp}.jpg")
            
            # Save the image
            image.save(image_path)
        
        # Get prediction
        score, recommendation = model.get_recommendation(
            image_path=image_path,  # Pass the saved image path
            temperature=temperature,
            humidity=humidity,
            date_str=date_str
        )
        
        # Clean up the temporary image
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        return {
            "score": float(score),
            "recommendation": recommendation,
            "data_used": {
                "image": image is not None,
                "temperature": temperature is not None,
                "humidity": humidity is not None,
                "date": date_str
            }
        }
        
    except Exception as e:
        # Clean up the temporary image in case of error
        if 'image_path' in locals() and image_path and os.path.exists(image_path):
            os.remove(image_path)
        return {"error": str(e)}

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Garden Image (Optional)"),
        gr.Number(label="Temperature (°F)"),
        gr.Number(label="Humidity (%)"),
        gr.Textbox(label="Date (YYYY-MM-DD)", optional=True)
    ],
    outputs=gr.JSON(label="Prediction Results"),
    title="Garden Recommender",
    description="Upload a garden image and provide environmental data to get a recommendation.",
    examples=[
        [None, 75, 60, "2024-03-20"],
        [None, 85, 45, "2024-07-15"]
    ]
)

if __name__ == "__main__":
    iface.launch() 