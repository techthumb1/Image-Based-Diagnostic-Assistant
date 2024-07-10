import logging
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC, Dice
from torchmetrics.segmentation import MeanIoU
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Segmentation metrics
dice = Dice(num_classes=1)
miou = MeanIoU(num_classes=1)

def dice_coefficient(pred, target):
    return dice(pred, target).item()

def intersection_over_union(pred, target):
    return miou(pred, target).item()

def pixel_accuracy(pred, target):
    return (pred == target).sum() / target.numel()

def mean_accuracy(pred, target):
    return pixel_accuracy(pred, target)

def mean_iou(pred, target):
    return miou(pred, target).item()

def boundary_f1_score(pred, target):
    from skimage.segmentation import find_boundaries
    from sklearn.metrics import f1_score as sklearn_f1_score
    pred_boundary = find_boundaries(pred.cpu().numpy(), mode='outer')
    target_boundary = find_boundaries(target.cpu().numpy(), mode='outer')
    return sklearn_f1_score(target_boundary.flatten(), pred_boundary.flatten())

# Classification metrics
accuracy_metric = Accuracy(task='multiclass', num_classes=3)
precision_metric = Precision(task='multiclass', num_classes=3)
recall_metric = Recall(task='multiclass', num_classes=3)
f1_metric = F1Score(task='multiclass', num_classes=3)
conf_matrix_metric = ConfusionMatrix(task='multiclass', num_classes=3)
auroc_metric = AUROC(task='multiclass', num_classes=3)

def calculate_metrics(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        avg_precision = average_precision_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)

        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"ROC AUC: {roc_auc}")
        logging.info(f"Average Precision: {avg_precision}")
        logging.info(f"Confusion Matrix: \n{conf_matrix}")
        logging.info(f"Accuracy: {accuracy}")
        
        return accuracy, precision, recall, f1, roc_auc, avg_precision, conf_matrix
    except Exception as e:
        logging.error(f"Error in calculate_metrics: {e}")
        return None, None, None, None, None, None, None

def compute_metrics(preds, labels, model_type):
    if model_type in ['segmentation', 'unetpp']:
        preds = preds.flatten()
        labels = labels.flatten()
    else:
        preds = torch.tensor([preds])
        labels = torch.tensor([labels])

    logger = logging.getLogger(__name__)    

    logger.info(f"Predictions shape: {preds.shape}, Labels shape: {labels.shape}")

    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=1)
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=1)
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=1)
    roc_auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')
    avg_precision = average_precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
    conf_matrix = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())

    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC AUC: {roc_auc}")
    logging.info(f"Average Precision: {avg_precision}")
    logging.info(f"Confusion Matrix: \n{conf_matrix}")
    logging.info(f"Accuracy: {accuracy}")
    
    return accuracy, precision, recall, f1, roc_auc, avg_precision, conf_matrix
