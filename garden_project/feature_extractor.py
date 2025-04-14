import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import io

class GardenFeatureExtractor:
    def __init__(self):
        # Load pretrained ResNet50 and remove the final classification layer
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract_features(self, image_input):
        """
        Extract features from an image using ResNet50
        Args:
            image_input: Either a path to an image file or image bytes
        Returns: 2048-dimensional feature vector
        """
        # Load and transform image
        if isinstance(image_input, str):
            # Handle file path
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, bytes):
            # Handle image bytes
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        else:
            raise ValueError("image_input must be either a file path or image bytes")
            
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
            
        # Flatten and convert to numpy array
        features = features.squeeze().numpy()
        
        # Calculate additional garden-specific features
        # These complement the ResNet features with domain-specific information
        additional_features = self._calculate_additional_features(image)
        
        # Combine ResNet and additional features
        combined_features = np.concatenate([features, additional_features])
        
        return combined_features
    
    def _calculate_additional_features(self, image):
        """
        Calculate additional garden-specific features
        Returns: Array of hand-crafted features
        """
        # Convert to numpy array for calculations
        img_array = np.array(image)
        
        # 1. Average brightness (indicator of good lighting)
        brightness = np.mean(img_array) / 255.0
        
        # 2. Color distribution (green for plants, blue for sky)
        if len(img_array.shape) == 3:  # Check if image is RGB
            green_ratio = np.mean(img_array[:, :, 1]) / 255.0
            blue_ratio = np.mean(img_array[:, :, 2]) / 255.0
        else:
            green_ratio = blue_ratio = 0.0
        
        # 3. Contrast (variety in the scene)
        contrast = np.std(img_array) / 255.0
        
        # 4. Color variance (variety of colors)
        if len(img_array.shape) == 3:
            color_variance = np.mean([np.std(img_array[:,:,i]) for i in range(3)]) / 255.0
        else:
            color_variance = 0.0
        
        return np.array([brightness, green_ratio, blue_ratio, contrast, color_variance])

    def get_feature_dim(self):
        """Return the dimension of the combined feature vector"""
        return 2048 + 5  # ResNet features + additional features 