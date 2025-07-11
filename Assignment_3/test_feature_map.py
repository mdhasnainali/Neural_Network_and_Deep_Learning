import cv2
import time
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50V2,
    MobileNet, MobileNetV2
)
from tensorflow.keras.applications import (
    vgg16, vgg19, resnet_v2,
    mobilenet, mobilenet_v2
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# MDS is available in sklearn
from sklearn.manifold import MDS

models_dict = {
    "VGG16": {
        "model": VGG16,
        "preprocess": vgg16.preprocess_input
    },
    "VGG19": {
        "model": VGG19,
        "preprocess": vgg19.preprocess_input
    },
    "ResNet50V2": {
        "model": ResNet50V2,
        "preprocess": resnet_v2.preprocess_input
    },
    "MobileNet": {
        "model": MobileNet,
        "preprocess": mobilenet.preprocess_input
    },
    "MobileNetV2": {
        "model": MobileNetV2,
        "preprocess": mobilenet_v2.preprocess_input
    },
}

# Dimensionality reduction methods
reduction_methods = {
    "PCA": {"method": PCA, "params": {"n_components": 2}},
    "t-SNE": {"method": TSNE, "params": {"n_components": 2, "random_state": 42, "perplexity": 30}},
    "MDS": {"method": MDS, "params": {"n_components": 2, "random_state": 42, "dissimilarity": 'euclidean'}},
    "LLE": {"method": LocallyLinearEmbedding, "params": {"n_components": 2, "n_neighbors": 10, "random_state": 42}},
    "Isomap": {"method": Isomap, "params": {"n_components": 2, "n_neighbors": 10}},
}


def load_and_generate_ds():
    """Load MNIST dataset and convert to RGB format for pre-trained models"""
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    
    # Convert grayscale to RGB and resize to 75x75
    def convert_to_rgb_and_resize(images):
        # Add channel dimension and convert to RGB
        images_rgb = np.stack([images] * 3, axis=-1)
        # Resize to 75x75
        resized_images = np.array([cv2.resize(img, (75, 75)) for img in images_rgb])
        return resized_images
    
    resized_train_x = convert_to_rgb_and_resize(train_x)
    resized_test_x = convert_to_rgb_and_resize(test_x)
    
    return (resized_train_x, train_y), (resized_test_x, test_y)


def get_feature_extractor(model_name):
    """Create a feature extractor by loading a model without the top classification layer"""
    backbone = models_dict[model_name]["model"](include_top=False, input_shape=(75,75,3))
    backbone.trainable = False
    # Add a global average pooling layer to get a flat feature vector
    inputs = backbone.input
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=inputs, outputs=x)
    return feature_extractor

def extract_features(feature_extractor, images, preprocess_fn):
    """Extract features from images using the feature extractor"""
    # Preprocess the images
    processed_images = preprocess_fn(images)
    # Extract features
    features = feature_extractor.predict(processed_images, verbose=0)
    return features

def reduce_features(features, method_name, method_info):
    """Reduce dimensionality of features using specified method"""
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Use sklearn methods
    reducer = method_info["method"](**method_info["params"])
    reduced_features = reducer.fit_transform(scaled_features)
    
    return reduced_features

def create_combined_visualization(model_name, all_reduced_features):
    """Create a single image with all dimensionality reduction methods in 2x5 grid"""
    # Check if we have features
    if len(all_reduced_features) < 1:
        print(f"No features collected for visualization for {model_name}")
        return
    
    # Create a figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Get method names in order
    available_methods = list(reduction_methods.keys())
    
    # Create class labels for MNIST
    class_names = [str(i) for i in range(10)]  # MNIST has digits 0-9
    
    # For each method, plot before (top row) and after (bottom row)
    for i, method_name in enumerate(available_methods):
        before_key = f"{method_name} - Before Training"
        after_key = f"{method_name} - After Training"
        
        # Plot before training (top row)
        if before_key in all_reduced_features:
            ax = axes[0, i]  # Top row, column i
            data = all_reduced_features[before_key]
            
            # Get unique labels and colors
            unique_labels = np.unique(data['labels'])
            colors = plt.cm.tab10(unique_labels)
            
            # Create scatter plot
            for j, label in enumerate(unique_labels):
                mask = data['labels'] == label
                ax.scatter(
                    data['features'][mask, 0],
                    data['features'][mask, 1],
                    color=colors[j],
                    label=class_names[label],
                    alpha=0.7,
                    s=15
                )
            
            # Add title and labels
            ax.set_title(f"{method_name}\nBefore Training", fontsize=12, fontweight='bold')
            ax.set_xlabel('Component 1', fontsize=10)
            ax.set_ylabel('Component 2', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Remove individual legends from subplots
            # We'll add a single legend at the bottom later
        
        # Plot after training (bottom row)
        if after_key in all_reduced_features:
            ax = axes[1, i]  # Bottom row, column i
            data = all_reduced_features[after_key]
            
            # Get unique labels and colors
            unique_labels = np.unique(data['labels'])
            colors = plt.cm.tab10(unique_labels)
            
            # Create scatter plot
            for j, label in enumerate(unique_labels):
                mask = data['labels'] == label
                ax.scatter(
                    data['features'][mask, 0],
                    data['features'][mask, 1],
                    color=colors[j],
                    label=class_names[label],
                    alpha=0.7,
                    s=15
                )
            
            # Add title and labels
            ax.set_title(f"After Training", fontsize=12, fontweight='bold')
            ax.set_xlabel('Component 1', fontsize=10)
            ax.set_ylabel('Component 2', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a single legend at the bottom for all plots
    # Get handles and labels from the first subplot that has data
    handles, labels = [], []
    for i, method_name in enumerate(available_methods):
        before_key = f"{method_name} - Before Training"
        if before_key in all_reduced_features:
            # Get legend information from this subplot
            ax = axes[0, i]
            h, l = ax.get_legend_handles_labels()
            if h and not handles:  # Only get handles once
                handles, labels = h, l
            break
    
    # Create a single legend at the bottom center of the figure
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                  fontsize=12, ncol=10, frameon=True, fancybox=True, shadow=True)
    
    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'Before Training', transform=axes[0, 0].transAxes, 
                    fontsize=14, fontweight='bold', rotation=90, va='center', ha='center')
    axes[1, 0].text(-0.15, 0.5, 'After Training', transform=axes[1, 0].transAxes, 
                    fontsize=14, fontweight='bold', rotation=90, va='center', ha='center')
    
    # Add a main title
    fig.suptitle(f'Dimensionality Reduction Methods: Feature Clustering Before vs After Training (MNIST) on {model_name}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0.02, 0.08, 1, 0.92])  # Adjusted bottom margin for legend
    plt.savefig(f'mnist_all_methods_comparison_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved as 'mnist_all_methods_comparison_{model_name}.png'")

def create_classifier_model(feature_extractor, num_classes):
    """Create a classifier model using the feature extractor as base"""
    inputs = feature_extractor.input
    x = feature_extractor.output
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_and_evaluate_with_all_methods(model_name, train_ds, test_ds):
    """Train a feature extractor and apply all dimensionality reduction methods"""
    print(f"\nProcessing model: {model_name}")
    
    # Initialize storage for this model's features
    model_reduced_features = {}
    
    # Extract data
    train_ds_x, train_ds_y = train_ds
    test_ds_x, test_ds_y = test_ds
    
    preprocess_fn = models_dict[model_name]["preprocess"]
    
    # Get feature extractor (backbone without head)
    feature_extractor = get_feature_extractor(model_name)
    
    # Sample 500 images from test set for better visualization (more than CIFAR-10 since MNIST is simpler)
    np.random.seed(42)
    indices = np.random.choice(len(test_ds_x), 500, replace=False)
    test_sample_x = test_ds_x[indices]
    test_sample_y = test_ds_y[indices].flatten()
    
    # Extract features before training
    print("Extracting features before training...")
    features_before = extract_features(feature_extractor, test_sample_x, preprocess_fn)
    
    # Apply all dimensionality reduction methods before training
    for method_name, method_info in reduction_methods.items():
        print(f"Applying {method_name} before training...")
        try:
            reduced_features_before = reduce_features(features_before, method_name, method_info)
            model_reduced_features[f"{method_name} - Before Training"] = {
                'features': reduced_features_before,
                'labels': test_sample_y
            }
        except Exception as e:
            print(f"Error applying {method_name} before training: {str(e)}")
            continue
    
    # Create a classifier on top of the feature extractor
    num_classes = len(np.unique(train_ds_y))
    model = create_classifier_model(feature_extractor, num_classes)
    
    # Make the feature extractor trainable for fine-tuning
    for layer in feature_extractor.layers:
        layer.trainable = True
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        loss=keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy']
    )
    
    # Train the model (reduced epochs for faster processing)
    print(f"Training {model_name}...")
    train_ds_x_processed = preprocess_fn(train_ds_x)
    model.fit(
        train_ds_x_processed, 
        train_ds_y, 
        epochs=1,  # Reduced for faster processing
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Extract features after training
    print("Extracting features after training...")
    features_after = extract_features(feature_extractor, test_sample_x, preprocess_fn)
    
    # Apply all dimensionality reduction methods after training
    for method_name, method_info in reduction_methods.items():
        print(f"Applying {method_name} after training...")
        try:
            reduced_features_after = reduce_features(features_after, method_name, method_info)
            model_reduced_features[f"{method_name} - After Training"] = {
                'features': reduced_features_after,
                'labels': test_sample_y
            }
        except Exception as e:
            print(f"Error applying {method_name} after training: {str(e)}")
            continue
    
    print(f"Completed processing {model_name}")
    
    # Create visualization for this model
    create_combined_visualization(model_name, model_reduced_features)
    
    # Clear memory
    del model, feature_extractor, model_reduced_features
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    # Create directories for outputs
    os.makedirs("dimensionality_reduction_plots", exist_ok=True)
    
    # Load MNIST dataset
    print("Loading and preprocessing MNIST dataset...")
    train_ds, test_ds = load_and_generate_ds()
    
    # Print dataset info
    print(f"Train dataset shape: {train_ds[0].shape}")
    print(f"Test dataset shape: {test_ds[0].shape}")
    print(f"Number of classes: {len(np.unique(train_ds[1]))}")
    
    # Process all 5 models to generate 5 figures
    models_to_process = ["VGG16", "VGG19", "ResNet50V2", "MobileNet", "MobileNetV2"]
    
    for model_name in models_to_process:
        try:
            print(f"\n{'='*50}")
            print(f"Starting processing for {model_name}")
            print(f"{'='*50}")
            
            # Train and apply all dimensionality reduction methods
            train_and_evaluate_with_all_methods(model_name, train_ds, test_ds)
            
            print(f"Successfully completed {model_name}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("Feature extraction and dimensionality reduction visualization completed!")
    print(f"Generated {len(models_to_process)} figures:")
    for model_name in models_to_process:
        print(f"- mnist_all_methods_comparison_{model_name}.png")
    print("="*60)
    
    print("\nAvailable dimensionality reduction methods:")
    for method in reduction_methods.keys():
        print(f"- {method}")