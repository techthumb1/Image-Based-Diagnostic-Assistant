import os
import numpy as np
from PIL import Image
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms

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
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

def load_and_preprocess_images_extended(img_dir, img_size=(224, 224)):
    images = []
    labels = []

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            file_path = os.path.join(root, file)
            img = None
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img = np.array(img)
            elif file.endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                img = dcm.pixel_array
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if img is not None:
                label = os.path.basename(root)
                images.append(img)
                labels.append(label)

    if len(images) != len(labels):
        raise ValueError("Number of images and labels do not match.")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def get_train_val_data(img_dir, img_size=(224, 224), test_size=0.2, random_state=42):
    images, labels = load_and_preprocess_images_extended(img_dir, img_size)
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size, random_state=random_state)
    return train_images, val_images, train_labels, val_labels
