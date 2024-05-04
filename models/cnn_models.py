import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('path_to_your_model.h5')
    return model

def model_predict(img_path):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img /= 255.0

    # Load your model
    model = load_model()
    preds = model.predict(img)
    return preds
