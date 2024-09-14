import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.6):
        super(EfficientNetCustom, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
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


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        confidence = 1.0 - self.smoothing
        log_preds = torch.log_softmax(preds, dim=-1)
        nll_loss = -log_preds.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)