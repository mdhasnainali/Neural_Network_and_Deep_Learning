import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import random

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class FloodSegmentationDataset:
    def __init__(self, image_dir, mask_dir, img_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.image_paths = sorted(glob(os.path.join(image_dir, "*")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*")))
        
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) don't match"
        
        print(f"Found {len(self.image_paths)} image-mask pairs")
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            return image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        return image
    
    def load_mask(self, mask_path):
        """Load and preprocess mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)  
        return mask
    
    def get_data(self):
        """Load all images and masks"""
        images = []
        masks = []
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            image = self.load_image(img_path)
            if image is None:
                continue
            mask = self.load_mask(mask_path)
            images.append(image)
            masks.append(mask)
        
        return np.array(images), np.array(masks)

def create_unet_model(input_shape=(256, 256, 3), num_classes=1):
    """Create U-Net model architecture"""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    
    # Decoder (Expansive Path)
    # Block 6
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(
        layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(conv9)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for binary segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union) metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Dice coefficient
    axes[0, 1].plot(history.history['dice_coefficient'], label='Training Dice')
    axes[0, 1].plot(history.history['val_dice_coefficient'], label='Validation Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Coefficient')
    axes[0, 1].legend()
    
    # IoU
    axes[1, 0].plot(history.history['iou_metric'], label='Training IoU')
    axes[1, 0].plot(history.history['val_iou_metric'], label='Validation IoU')
    axes[1, 0].set_title('IoU Metric')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    
    # Accuracy
    axes[1, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=4):
    """Visualize model predictions"""
    predictions = model.predict(X_test[:num_samples])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_mask = (predictions[i] > 0.5).astype(np.uint8)
        axes[i, 2].imshow(pred_mask.squeeze(), cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    IMAGE_DIR = "./archive/Image/"
    MASK_DIR = "./archive/Mask"
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Check if directories exist
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory '{IMAGE_DIR}' not found!")
        return
    if not os.path.exists(MASK_DIR):
        print(f"Error: Mask directory '{MASK_DIR}' not found!")
        return
    
    # Load dataset
    print("Loading dataset...")
    dataset = FloodSegmentationDataset(IMAGE_DIR, MASK_DIR, IMG_SIZE)
    X, y = dataset.get_data()
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Image range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Mask range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    print("Creating U-Net model...")
    model = create_unet_model(input_shape=(*IMG_SIZE, 3), num_classes=1)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=dice_loss,
        metrics=[dice_coefficient, iou_metric, 'accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_unet_flood_model.keras',
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=10,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    best_model = keras.models.load_model(
        'best_unet_flood_model.keras',
        custom_objects={
            'dice_loss': dice_loss,
            'dice_coefficient': dice_coefficient,
            'iou_metric': iou_metric
        }
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Dice Coefficient: {test_results[1]:.4f}")
    print(f"Test IoU: {test_results[2]:.4f}")
    print(f"Test Accuracy: {test_results[3]:.4f}")
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(best_model, X_test, y_test)
    
    # Save final model
    best_model.save('final_unet_flood_model.h5')
    print("Model saved as 'final_unet_flood_model.h5'")

if __name__ == "__main__":
    main()

