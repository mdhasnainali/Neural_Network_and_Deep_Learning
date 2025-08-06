import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, datasets
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Load and Preprocess Data (CIFAR-10) ---
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Preprocessing for ImageNet pre-trained models:
# ImageNet models expect input shape (32, 32, 3) or similar, and specific preprocessing.
# CIFAR-10 images are (32, 32, 3). We need to resize them.
# Note: Resizing small images to large ones can lead to blurring and loss of detail,
# which might affect performance compared to models trained on larger native images.
# However, for demonstrating fine-tuning, this approach works.

# Resize images to 32x32 and normalize
train_images_resized = tf.image.resize(train_images, (32, 32))
test_images_resized = tf.image.resize(test_images, (32, 32))

# Convert to float32 and apply VGG-specific preprocessing (mean subtraction from ImageNet)
# For ResNet, typically scaling to [0, 1] then using preprocess_input is sufficient,
# but VGG's specific preprocessing is often applied. For simplicity, we'll use
# applications.preprocess_input which is general.
train_images_processed = applications.vgg16.preprocess_input(train_images_resized)
test_images_processed = applications.vgg16.preprocess_input(test_images_resized)

# One-hot encode labels
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

print(f"Dataset loaded and preprocessed. Training images shape: {train_images_processed.shape}")

# --- 2. Fine-tuning VGG16 ---
print("\n--- Fine-tuning VGG16 ---")
# Load VGG16 pre-trained on ImageNet, without the top classification layer
base_vgg16 = applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the convolutional base
base_vgg16.trainable = False

# Create a new model on top
inputs = tf.keras.Input(shape=(32, 32, 3))
x = base_vgg16(inputs, training=False) # Important: set training=False when using frozen base
x = layers.GlobalAveragePooling2D()(x) # Use GlobalAveragePooling for VGG
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x) # Add dropout for regularization
outputs = layers.Dense(10, activation='softmax')(x) # 10 classes

vgg16_finetuned_model = models.Model(inputs, outputs)

# Compile the model
vgg16_finetuned_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), # Lower LR for fine-tuning
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

vgg16_finetuned_model.summary()

print("Training VGG16 fine-tuned model (only top layers)...")
history_vgg16 = vgg16_finetuned_model.fit(train_images_processed, train_labels_one_hot,
                                         epochs=5, # Short training for demonstration
                                         validation_data=(test_images_processed, test_labels_one_hot),
                                         batch_size=32)

print("\nEvaluating VGG16 fine-tuned model...")
test_loss_vgg16, test_acc_vgg16 = vgg16_finetuned_model.evaluate(test_images_processed, test_labels_one_hot, verbose=2)
print(f"Test accuracy of VGG16 fine-tuned: {test_acc_vgg16:.4f}")

# Save VGG16 fine-tuned model
vgg16_model_save_path = 'vgg16_finetuned.keras'
print(f"Saving VGG16 fine-tuned model to: {vgg16_model_save_path}")
vgg16_finetuned_model.save(vgg16_model_save_path)
print("VGG16 fine-tuned model saved successfully.")

# --- 3. Fine-tuning ResNet50 ---
print("\n--- Fine-tuning ResNet50 ---")
# Load ResNet50 pre-trained on ImageNet, without the top classification layer
base_resnet50 = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the convolutional base
base_resnet50.trainable = False

# Create a new model on top
inputs = tf.keras.Input(shape=(32, 32, 3))
x = base_resnet50(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x) # ResNet typically uses GlobalAveragePooling
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x) # 10 classes

resnet50_finetuned_model = models.Model(inputs, outputs)

# Compile the model
resnet50_finetuned_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

resnet50_finetuned_model.summary()

print("Training ResNet50 fine-tuned model (only top layers)...")
history_resnet50 = resnet50_finetuned_model.fit(train_images_processed, train_labels_one_hot,
                                               epochs=5, # Short training for demonstration
                                               validation_data=(test_images_processed, test_labels_one_hot),
                                               batch_size=32)

print("\nEvaluating ResNet50 fine-tuned model...")
test_loss_resnet50, test_acc_resnet50 = resnet50_finetuned_model.evaluate(test_images_processed, test_labels_one_hot, verbose=2)
print(f"Test accuracy of ResNet50 fine-tuned: {test_acc_resnet50:.4f}")

# Save ResNet50 fine-tuned model
resnet50_model_save_path = 'resnet50_finetuned.keras'
print(f"Saving ResNet50 fine-tuned model to: {resnet50_model_save_path}")
resnet50_finetuned_model.save(resnet50_model_save_path)
print("ResNet50 fine-tuned model saved successfully.")