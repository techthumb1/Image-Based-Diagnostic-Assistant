import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load pre-trained EfficientNet model for classification
def get_efficientnet_model():
    try:
        logger.info("Attempting to load EfficientNet model...")
        model = EfficientNet.from_pretrained('efficientnet-b0')
        logger.info("EfficientNet model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load EfficientNet model: {e}")
        raise

# Load pre-trained Segformer model for segmentation
def get_segmentation_model():
    try:
        logger.info("Attempting to load Segformer model...")
        model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        logger.info("Segformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load Segformer model: {e}")
        raise

# Load pre-trained Segformer feature extractor
def get_feature_extractor():
    try:
        logger.info("Attempting to load Segformer feature extractor...")
        feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        logger.info("Feature extractor loaded successfully")
        return feature_extractor
    except Exception as e:
        logger.error(f"Failed to load Segformer feature extractor: {e}")
        raise

# Load pre-trained UNet++ model for segmentation
def get_unetpp_model():
    try:
        logger.info("Attempting to load UNet++ model...")
        model_unetpp = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",  # Set to None because you will load the weights separately
            in_channels=3,
            classes=2
        )
        logger.info("UNet++ model loaded successfully")
        return model_unetpp
    except Exception as e:
        logger.error(f"Failed to load UNet++ model: {e}")
        raise

# Load pre-trained classification model
def get_classification_model(num_classes):
    try:
        logger.info("Attempting to load EfficientNet model for classification...")
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, num_classes)  # Update the final layer to match num_classes
        logger.info("EfficientNet model for classification loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load EfficientNet model for classification: {e}")
        raise

# Define the optimizer
def get_unetpp_optimizer(model, learning_rate=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer

# Training loop for classification model
def train_classification_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        logger.info(f'Validation Loss: {val_loss/len(val_loader)}')

    return model

# Training loop for segmentation model
def train_segmentation_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range (num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        logger.info(f'Validation Loss: {val_loss/len(val_loader)}')

    return model

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model
def load_model(path, model_type='classification', num_classes=10):
    if model_type == 'classification':
        model = get_classification_model(num_classes)
    elif model_type == 'segmentation':
        model = get_segmentation_model()
    elif model_type == 'unetpp':
        model = get_unetpp_model()
    model.load_state_dict(torch.load(path))
    return model

# Evaluate model
def evaluate_model(model, dataloader, model_type='classification'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            if model_type == 'classification':
                preds = outputs.argmax(dim=1)
            elif model_type == 'segmentation' or model_type == 'unetpp':
                preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds
