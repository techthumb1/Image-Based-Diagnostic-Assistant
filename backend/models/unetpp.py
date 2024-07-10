import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.optim as optim

class UNetPP(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetPP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def load_and_preprocess_image(image_path, transform):
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply transforms
    image_tensor = transform(image)
    #image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def model_predict(image_tensor, model, feature_extractor):
    if not feature_extractor:
        raise ValueError("Feature extractor is not initialized.")

    # Use the feature extractor to process the image array
    inputs = feature_extractor(images=image_tensor, return_tensors="pt")
    pixel_values = inputs['pixel_values']

    if pixel_values.shape[0] == 1:
        pixel_values = torch.cat([pixel_values, pixel_values], dim=0)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits
    predicted_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
    return predicted_mask

def train_unetpp_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            if inputs.size(0) == 1:  # Skip batches with only one element
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.size(0) == 1:  # Skip batches with only one element
                    continue
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f'Validation Loss: {val_loss/len(val_loader)}')

    return model
