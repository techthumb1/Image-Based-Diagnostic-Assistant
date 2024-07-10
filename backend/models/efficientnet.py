import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.6):
        super(EfficientNetCustom, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.BatchNorm1d(num_ftrs // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs // 2, num_classes)
        )

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.model._fc(x)
        return x

def get_classification_model(num_classes, dropout_rate=0.6):
    return EfficientNetCustom(num_classes, dropout_rate)
