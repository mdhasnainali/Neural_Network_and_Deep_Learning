import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2
import scipy.io
from scipy.ndimage import gaussian_filter
import glob
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MallDatasetLoader:
    """
    Loads and preprocesses the Mall dataset for crowd counting
    """
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.frames_path = self.data_path / "frames"
        self.gt_path = self.data_path / "mall_gt.mat"
        
    def load_ground_truth(self):
        """Load ground truth annotations from mall_gt.mat"""
        gt_data = scipy.io.loadmat(str(self.gt_path))
        # Extract count for each frame
        frame_counts = gt_data['count'].flatten()
        return frame_counts
    
    def load_frames(self):
        """Load all frame images"""
        frame_files = sorted(glob.glob(str(self.frames_path / "*.jpg")))
        frames = []
        
        for frame_file in frame_files:
            img = cv2.imread(frame_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        return np.array(frames)
    
    def create_density_map(self, img_shape, count):
        """
        Create a simplified density map for the Mall dataset
        Since we only have count labels, we'll create a uniform density map
        """
        h, w = img_shape[:2]
        # Create a simple gaussian density map centered in the image
        density_map = np.zeros((h, w))
        
        if count > 0:
            # Add random points and blur them to create density
            num_points = max(1, int(count))
            for _ in range(num_points):
                x = np.random.randint(w//4, 3*w//4)
                y = np.random.randint(h//4, 3*h//4)
                density_map[y, x] = 1
            
            # Apply Gaussian blur
            density_map = gaussian_filter(density_map, sigma=2)
            
            # Normalize to match the count
            if density_map.sum() > 0:
                density_map = density_map * count / density_map.sum()
        
        return density_map

def build_mcnn(input_shape):
    """
    Build Multi-Column CNN architecture for crowd counting
    """
    inputs = keras.Input(shape=input_shape)
    
    # Column 1: Small receptive field (for close-up heads)
    conv1_1 = layers.Conv2D(16, (9, 9), activation='relu', padding='same')(inputs)
    conv1_2 = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(conv1_1)
    conv1_3 = layers.Conv2D(16, (7, 7), activation='relu', padding='same')(conv1_2)
    conv1_4 = layers.Conv2D(8, (7, 7), activation='relu', padding='same')(conv1_3)
    
    # Column 2: Medium receptive field (for medium distance heads)
    conv2_1 = layers.Conv2D(20, (7, 7), activation='relu', padding='same')(inputs)
    conv2_2 = layers.Conv2D(40, (5, 5), activation='relu', padding='same')(conv2_1)
    conv2_3 = layers.Conv2D(20, (5, 5), activation='relu', padding='same')(conv2_2)
    conv2_4 = layers.Conv2D(10, (5, 5), activation='relu', padding='same')(conv2_3)
    
    # Column 3: Large receptive field (for far distance heads)
    conv3_1 = layers.Conv2D(24, (5, 5), activation='relu', padding='same')(inputs)
    conv3_2 = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = layers.Conv2D(24, (3, 3), activation='relu', padding='same')(conv3_2)
    conv3_4 = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(conv3_3)
    
    # Concatenate all columns
    merged = layers.Concatenate()([conv1_4, conv2_4, conv3_4])
    
    # Final layers
    conv_final = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(merged)
    
    model = keras.Model(inputs=inputs, outputs=conv_final)
    return model

def preprocess_data(images, counts, target_size=(224, 224)):
    """
    Preprocess images and create density maps
    """
    processed_images = []
    density_maps = []
    
    for i, (img, count) in enumerate(zip(images, counts)):
        # Resize image
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Create density map
        density_map = MallDatasetLoader('').create_density_map(target_size + (3,), count)
        density_map = cv2.resize(density_map, target_size)
        
        processed_images.append(img_normalized)
        density_maps.append(density_map)
    
    return np.array(processed_images), np.array(density_maps)

def split_dataset(X, y, test_size=0.2, val_size=0.2):
    """
    Split dataset into train, validation, and test sets
    """
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Second split: separate train and validation from the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def custom_loss(y_true, y_pred):
    """
    Custom loss function for density map regression
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def count_loss(y_true, y_pred):
    """
    Loss function based on total count
    """
    count_true = tf.reduce_sum(y_true, axis=[1, 2])
    count_pred = tf.reduce_sum(y_pred, axis=[1, 2])
    return tf.reduce_mean(tf.square(count_true - count_pred))

def main():
    """
    Main training and evaluation pipeline
    """
    # Configuration
    DATA_PATH = "./mall_dataset"  # Update this path
    TARGET_SIZE = (224, 224)
    BATCH_SIZE = 64
    EPOCHS = 2
    
    print("Loading Mall Dataset...")
    
    # Load dataset
    loader = MallDatasetLoader(DATA_PATH)
    
    try:
        images = loader.load_frames()
        counts = loader.load_ground_truth()
        
        print(f"Loaded {len(images)} images with counts")
        print(f"Count range: {counts.min()} - {counts.max()}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the mall dataset is in the correct directory structure")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(images, counts, TARGET_SIZE)
    
    # Split dataset
    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build model
    print("Building MCNN model...")
    model = build_mcnn(input_shape=TARGET_SIZE + (3,))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=[count_loss]
    )
    
    print("Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_mcnn_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Reshape density maps for training
    y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
    y_val_reshaped = y_val.reshape(y_val.shape[0], y_val.shape[1], y_val.shape[2], 1)
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train_reshaped,
        validation_data=(X_val, y_val_reshaped),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate count predictions
    test_counts_true = np.sum(y_test_reshaped, axis=(1, 2, 3))
    test_counts_pred = np.sum(y_pred, axis=(1, 2, 3))
    
    # Calculate metrics
    mae = mean_absolute_error(test_counts_true, test_counts_pred)
    mse = mean_squared_error(test_counts_true, test_counts_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nTest Results:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_counts_true, test_counts_pred, alpha=0.6)
    plt.plot([test_counts_true.min(), test_counts_true.max()], 
             [test_counts_true.min(), test_counts_true.max()], 'r--', lw=2)
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('True vs Predicted Counts')
    
    plt.tight_layout()
    plt.savefig('training_history_mcnn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history plot saved as 'training_history_mcnn.png'")
    
    # Show some sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(X_test))):
        true_count = test_counts_true[i]
        pred_count = test_counts_pred[i]
        print(f"Sample {i+1}: True={true_count:.1f}, Predicted={pred_count:.1f}, Error={abs(true_count-pred_count):.1f}")

if __name__ == "__main__":
    main()
