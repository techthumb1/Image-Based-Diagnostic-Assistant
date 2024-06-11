import os
import torch
from torch.utils.data import DataLoader  # Add this line
from transformers import SegformerForImageClassification, SegformerImageProcessor
from backend.models.model_training import load_data, get_model, train_model, save_model

# Define paths
model_save_path = 'backend/models/segformer_pruned_model/pytorch_model.bin'
data_dir = 'input/images' 
feature_extractor_path = 'backend/models/segformer_feature_extractor' 
num_classes = 10  # Replace with your number of classes

# Load data
dataloader = load_data(data_dir, batch_size=32)

# Load feature extractor
feature_extractor = SegformerImageProcessor.from_pretrained(feature_extractor_path)

# Get model
model = get_model(num_classes)

# Preprocess images
def preprocess_data(dataloader, feature_extractor):
    processed_data = []
    for images, labels in dataloader:
        inputs = feature_extractor(images=images, return_tensors="pt")
        processed_data.append((inputs['pixel_values'], labels))
    return processed_data

# Preprocess data
processed_data = preprocess_data(dataloader, feature_extractor)

# Convert processed data to DataLoader
processed_dataloader = DataLoader(processed_data, batch_size=32, shuffle=True)

# Train model
trained_model = train_model(model, processed_dataloader, num_epochs=10)

# Save model
save_model(trained_model, model_save_path)
print(f"Model saved at {model_save_path}")
