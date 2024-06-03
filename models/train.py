from models.models import create_model, augment_images
from model import create_model
from image_loader.image_load import augment_images

def train_model():
    model = create_model()
    train_generator = augment_images('input/images')
    model.fit(train_generator, epochs=10)
    model.save('models/model.h5')
