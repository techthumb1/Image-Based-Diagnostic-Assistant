import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        conf_matrix = confusion_matrix(y_true, y_pred)

        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Confusion Matrix: \n{conf_matrix}")
        
        return accuracy, precision, recall, f1, conf_matrix
    except Exception as e:
        logging.error(f"Error in log_metrics: {e}")
        return None, None, None, None


# Example usage
y_true = [0, 1, 2, 2, 0]  # Replace with actual true labels
y_pred = [0, 1, 2, 1, 0]  # Replace with actual model predictions

accuracy, precision, recall, f1, conf_matrix = calculate_metrics(y_true, y_pred)
