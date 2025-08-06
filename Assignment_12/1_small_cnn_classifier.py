import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Load and Preprocess Data (CIFAR-10) ---
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode labels
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Dataset loaded. Training images shape: {train_images.shape}, Test images shape: {test_images.shape}")
print(f"Number of classes: {len(class_names)}")

# --- 2. Define the Small CNN Model ---
print("Defining the Small CNN model...")
small_cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 classes for CIFAR-10
])

# Print model summary
small_cnn_model.summary()

# --- 3. Compile the Model ---
print("Compiling the model...")
small_cnn_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# --- 4. Train the Model ---
print("Training the model...")
history = small_cnn_model.fit(train_images, train_labels_one_hot,
                               epochs=10, # You can increase epochs for better performance
                               validation_data=(test_images, test_labels_one_hot),
                               batch_size=64)

# --- 5. Evaluate the Model ---
print("\nEvaluating the trained model...")
test_loss, test_acc = small_cnn_model.evaluate(test_images, test_labels_one_hot, verbose=2)
print(f"Test accuracy of Small CNN: {test_acc:.4f}")

# --- 6. Save the Model ---
model_save_path = 'small_cnn_model.keras'
print(f"Saving the Small CNN model to: {model_save_path}")
small_cnn_model.save(model_save_path)
print("Small CNN model saved successfully.")