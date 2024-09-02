import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from segmentation_models_pytorch import UnetPlusPlus
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.encoded_labels[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label

# Data augmentation and preprocessing
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

# Define the model
model_unetpp = UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_unetpp.to(device)

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def preprocess_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define a prediction function
def predict(image_tensor, model):
    model.eval()
    with torch.no_grad():
        # Ensure batch size is more than 1
        if image_tensor.size(0) == 1:  # If batch size is 1, duplicate to create a batch size of 2
            image_tensor = torch.cat([image_tensor, image_tensor], dim=0)
        output = model(image_tensor)
        # If batch size was artificially increased, revert the batch size for output
        if output.size(0) == 2:
            output = output[:1]
        return output
    