import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MallDatasetLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.frames_path = os.path.join(data_path, 'frames')
        self.gt_path = os.path.join(data_path, 'mall_gt.mat')
        
    def load_data(self):
        """Load mall dataset images and ground truth"""
        # Load ground truth data
        gt_data = loadmat(self.gt_path)
        gt_counts = gt_data['count'].flatten()
        
        # Load images
        images = []
        frame_files = sorted([f for f in os.listdir(self.frames_path) if f.endswith(('.jpg', '.png'))])
        
        for i, frame_file in enumerate(frame_files):
            if i < len(gt_counts):  # Ensure we have corresponding ground truth
                img_path = os.path.join(self.frames_path, frame_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                else:
                    print(f"Warning: Could not load image {frame_file}")
        
        # Trim gt_counts to match loaded images
        gt_counts = gt_counts[:len(images)]
        
        print(f"Loaded {len(images)} images with ground truth counts")
        print(f"Count range: {np.min(gt_counts)} - {np.max(gt_counts)}")
        
        return np.array(images), np.array(gt_counts)

class DataPreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
    def preprocess_images(self, images):
        """Resize and normalize images"""
        processed_images = []
        for img in images:
            # Resize image
            resized = cv2.resize(img, self.target_size)
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            processed_images.append(normalized)
        
        return np.array(processed_images)
    
    def create_density_maps(self, images, counts):
        """Create density maps from count labels"""
        density_maps = []
        
        for i, count in enumerate(counts):
            # Create a simple density map
            h, w = self.target_size
            density_map = np.zeros((h, w, 1), dtype=np.float32)
            
            # For simplicity, create a Gaussian distribution centered in the image
            # In practice, you'd want to use actual person locations if available
            if count > 0:
                # Create random points for crowd distribution
                num_points = min(int(count), 50)  # Limit for computational efficiency
                for _ in range(num_points):
                    x = np.random.randint(20, w-20)
                    y = np.random.randint(20, h-20)
                    
                    # Create Gaussian blob
                    y_grid, x_grid = np.ogrid[:h, :w]
                    sigma = 8
                    gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
                    density_map[:, :, 0] += gaussian * (count / num_points)
            
            density_maps.append(density_map)
        
        return np.array(density_maps)

def build_unet(input_size=(256, 256, 3)):
    """Build U-Net architecture for crowd counting"""
    inputs = keras.Input(shape=input_size)
    
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
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder (Expansive Path)
    # Block 6
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer - single channel for density map
    outputs = layers.Conv2D(1, 1, activation='relu', padding='same')(conv9)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def mean_absolute_error_count(y_true, y_pred):
    """Custom metric for counting accuracy"""
    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean(tf.abs(true_count - pred_count))

def root_mean_squared_error_count(y_true, y_pred):
    """Custom metric for counting RMSE"""
    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.sqrt(tf.reduce_mean(tf.square(true_count - pred_count)))

class CrowdCountingTrainer:
    def __init__(self, model, data_path):
        self.model = model
        self.data_path = data_path
        self.history = None
        
    def prepare_data(self):
        """Load and prepare dataset"""
        print("Loading mall dataset...")
        loader = MallDatasetLoader(self.data_path)
        images, counts = loader.load_data()
        
        print("Preprocessing data...")
        preprocessor = DataPreprocessor()
        processed_images = preprocessor.preprocess_images(images)
        density_maps = preprocessor.create_density_maps(images, counts)
        
        # Split data into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_images, density_maps, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, epochs=5, batch_size=32):
        """Train the U-Net model"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data()
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[mean_absolute_error_count, root_mean_squared_error_count]
        )
        
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
                'best_unet_crowd_counting.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_mae, test_rmse = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE (Count): {test_mae:.4f}")
        print(f"Test RMSE (Count): {test_rmse:.4f}")
        
        return self.history, (X_test, y_test)
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mean_absolute_error_count'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mean_absolute_error_count'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error (Count)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # RMSE
        axes[1, 0].plot(self.history.history['root_mean_squared_error_count'], label='Training RMSE')
        axes[1, 0].plot(self.history.history['val_root_mean_squared_error_count'], label='Validation RMSE')
        axes[1, 0].set_title('Root Mean Squared Error (Count)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('training_history_unet.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training history plot saved as 'training_history.png'")
    
    def visualize_predictions(self, test_data, num_samples=4):
        """Visualize model predictions"""
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.model.predict(X_test[:num_samples])
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(X_test[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth density map
            axes[i, 1].imshow(y_test[i][:, :, 0], cmap='hot')
            true_count = np.sum(y_test[i])
            axes[i, 1].set_title(f'Ground Truth (Count: {true_count:.1f})')
            axes[i, 1].axis('off')
            
            # Predicted density map
            axes[i, 2].imshow(predictions[i][:, :, 0], cmap='hot')
            pred_count = np.sum(predictions[i])
            axes[i, 2].set_title(f'Prediction (Count: {pred_count:.1f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions_visualization_ucnn.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Predictions visualization saved as 'predictions_visualization.png'")

# Main execution
if __name__ == "__main__":
    # Set your dataset path here
    DATA_PATH = "./mall_dataset"  # Change this to your actual dataset path
    
    # Check if dataset path exists
    if not os.path.exists(DATA_PATH):
        print(f"Dataset path not found: {DATA_PATH}")
        print("Please update the DATA_PATH variable with the correct path to your mall dataset.")
        exit(1)
    
    print("Building U-Net model...")
    model = build_unet()
    model.summary()
    
    # Initialize trainer
    trainer = CrowdCountingTrainer(model, DATA_PATH)
    
    # Train model
    history, test_data = trainer.train(epochs=5, batch_size=32)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Visualize predictions
    trainer.visualize_predictions(test_data, num_samples=4)
    
    print("\nTraining completed! Model saved as 'best_unet_crowd_counting.h5'")