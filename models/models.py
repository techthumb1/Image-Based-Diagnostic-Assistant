import tensorflow as tf
import tensorflow_hub as hub

# Create a model using the MobileNetV2 architecture
def create_model():
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],
                       trainable=False),  # Can be set to trainable=True to fine-tune
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Load a model from a saved file
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# Save a model to a file
def save_model(model, model_path):
    model.save(model_path)


# Load a configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Save a configuration file
def save_config(config, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


# Load an image from a file
def load_image(image_path):
    return tf.io.read_file(image, image_path)
