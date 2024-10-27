import os
from PIL import Image
import numpy as np
import pydicom
import cv2
import logging

logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    try:
        if image_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.dcm')):
            if image_path.endswith('.dcm'):
                dcm = pydicom.dcmread(image_path)
                img = dcm.pixel_array
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(img_size)
                img = np.array(img)
        else:
            raise ValueError(f"Unsupported file format: {image_path}")

        logger.info(f"Image loaded and preprocessed: {image_path}")
        return img
    except Exception as e:
        logger.error(f"Error in loading and preprocessing image: {e}")
        raise

