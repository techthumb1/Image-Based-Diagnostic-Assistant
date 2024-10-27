from transformers import ViTModel, ViTFeatureExtractor
import tensorflow as tf

def create_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = model(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(features)
    
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
