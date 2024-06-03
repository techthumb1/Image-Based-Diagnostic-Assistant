import tensorflow as tf

def preprocess_images(file_content):
    image = tf.image.decode_jpeg(file_content, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return tf.expand_dims(image, 0)
