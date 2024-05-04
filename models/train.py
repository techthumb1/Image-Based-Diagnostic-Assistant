from model import create_model, augment_images

def train_model():
    model = create_model()
    train_generator = augment_images('path_to_train_data')
    model.fit(train_generator, epochs=10)
    model.save('path_to_save_model/model.h5')
