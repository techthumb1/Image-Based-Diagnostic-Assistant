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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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

#def load_and_preprocess_images(img_dir, img_size=(224, 224)):
#    images = []
#    labels = []
#
#    for root, dirs, files in os.walk(img_dir):
#        for file in files:
#            file_path = os.path.join(root, file)
#            img = None
#            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#                img = Image.open(file_path).convert('RGB')
#                img = img.resize(img_size)
#                img = np.array(img)
#            elif file.endswith('.dcm'):
#                dcm = pydicom.dcmread(file_path)
#                img = dcm.pixel_array
#                img = cv2.resize(img, img_size)
#                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#            if img is not None:
#                label = os.path.basename(root)
#                images.append(img)
#                labels.append(label)
#
#    if len(images) != len(labels):
#        raise ValueError("Number of images and labels do not match.")
#
#    images = np.array(images)
#    labels = np.array(labels)
#
#    return images, labels

#def get_train_val_data(img_dir, img_size=(224, 224), test_size=0.2, random_state=42):
#    images, labels = load_and_preprocess_images(img_dir, img_size)
#    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size, random_state=random_state)
#    return train_images, val_images, train_labels, val_labels
#