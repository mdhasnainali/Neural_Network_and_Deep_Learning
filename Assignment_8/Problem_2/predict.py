import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

# === Load model ===
model = tf.keras.models.load_model(
    'best_unet_crowd_counting.keras',
    custom_objects={
        'mean_absolute_error_count': lambda y_true, y_pred: tf.reduce_mean(tf.abs(tf.reduce_sum(y_true, axis=[1, 2, 3]) - tf.reduce_sum(y_pred, axis=[1, 2, 3]))),
        'root_mean_squared_error_count': lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(tf.reduce_sum(y_true, axis=[1, 2, 3]) - tf.reduce_sum(y_pred, axis=[1, 2, 3]))))
    }
)

# === Load and preprocess the input image ===
def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # Add batch dimension

# === Save predicted density map ===
def save_density_map(density_map, output_path):
    density_map = density_map[0, :, :, 0]  # Remove batch and channel dims
    plt.imsave(output_path, density_map, cmap='hot')
    print(f"Saved density map to: {output_path}")

# === Predict and save ===
input_image_path = "mall_dataset/frames/seq_000001.jpg"
output_map_path = 'predicted_density_map.png'

img = preprocess_image(input_image_path)
pred = model.predict(img)

save_density_map(pred, output_map_path)
