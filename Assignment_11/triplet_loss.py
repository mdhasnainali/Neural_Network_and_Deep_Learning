import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model, optimizers, metrics
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from typing import Tuple, List, Optional


# Configuration
CONFIG = {
    'epochs': 20,
    'batch_size': 16,
    'margin': 1.0,
    'target_shape': (128, 128),
    'embedding_dim': 256,
    'learning_rate': 0.0001,
    'train_split': 0.8
}

# File paths
PATHS = {
    'lfw_dir': "LFW/",
    'image_dir': "LFW/lfw-deepfunneled",
    'pairs_file': "LFW/pairs.csv"
}


def load_and_preprocess_image(image_path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """Load and preprocess a single image."""
    try:
        image = Image.open(image_path).convert("RGB").resize(target_shape)
        return np.array(image, dtype="float32")
    except Exception as e:
        raise FileNotFoundError(f"Could not load image {image_path}: {e}")


def get_all_image_paths(image_dir: str) -> List[str]:
    """Get all image paths from the directory."""
    image_paths = []
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


def parse_pairs_file(pairs_path: str) -> List[Tuple[str, str, str]]:
    """Parse the LFW pairs file to extract positive pairs."""
    positive_pairs = []
    
    with open(pairs_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                person = parts[0]
                img_num1, img_num2 = int(parts[1]), int(parts[2])
                
                path1 = os.path.join(PATHS['image_dir'], person, f"{person}_{img_num1:04d}.jpg")
                path2 = os.path.join(PATHS['image_dir'], person, f"{person}_{img_num2:04d}.jpg")
                
                positive_pairs.append((path1, path2, person))
    
    return positive_pairs


def create_triplet_dataset(image_dir: str, pairs_path: str, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create triplet dataset from LFW pairs."""
    print("Loading image paths...")
    all_image_paths = get_all_image_paths(image_dir)
    
    print("Parsing pairs file...")
    positive_pairs = parse_pairs_file(pairs_path)
    
    print(f"Found {len(positive_pairs)} positive pairs. Creating triplets...")
    
    anchors, positives, negatives = [], [], []
    
    for anchor_path, positive_path, person_name in positive_pairs:
        try:
            # Load anchor and positive images
            anchor_img = load_and_preprocess_image(anchor_path, target_shape)
            positive_img = load_and_preprocess_image(positive_path, target_shape)
            
            # Find negative image from different person
            negative_path = random.choice(all_image_paths)
            while person_name in negative_path:
                negative_path = random.choice(all_image_paths)
            
            negative_img = load_and_preprocess_image(negative_path, target_shape)
            
            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)
            
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping triplet.")
            continue
    
    print(f"Successfully created {len(anchors)} triplets.")
    return np.array(anchors), np.array(positives), np.array(negatives)


def shuffle_triplets(anchors: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle triplets while maintaining correspondence."""
    indices = np.random.permutation(len(anchors))
    return anchors[indices], positives[indices], negatives[indices]


def split_dataset(anchors: np.ndarray, positives: np.ndarray, negatives: np.ndarray, 
                 split_ratio: float = 0.8) -> Tuple[np.ndarray, ...]:
    """Split dataset into training and validation sets."""
    split_index = int(split_ratio * len(anchors))
    
    train_data = (
        anchors[:split_index],
        positives[:split_index], 
        negatives[:split_index]
    )
    
    val_data = (
        anchors[split_index:],
        positives[split_index:],
        negatives[split_index:]
    )
    
    return train_data + val_data


def preprocess_triplets(*triplets: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Apply ResNet preprocessing to triplets."""
    return tuple(preprocess_input(triplet.copy()) for triplet in triplets)


def create_embedding_network(input_shape: Tuple[int, int, int], embedding_dim: int = 256) -> Model:
    """Create the embedding network using ResNet50V2."""
    # Base ResNet model
    base_model = ResNet50V2(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False
    )
    
    # Fine-tune from conv5_block1_out layer
    trainable = False
    for layer in base_model.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable
    
    # Add embedding head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    embeddings = layers.Dense(embedding_dim, activation='linear')(x)
    
    # L2 normalize embeddings
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    
    return Model(inputs=base_model.input, outputs=embeddings, name="EmbeddingNetwork")


def triplet_loss(y_true, y_pred, margin: float = 1.0):
    """Triplet loss function."""
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    loss = tf.maximum(positive_dist - negative_dist + margin, 0.0)
    return tf.reduce_mean(loss)


def create_triplet_model(input_shape: Tuple[int, int, int], embedding_dim: int = 256, margin: float = 1.0) -> Model:
    """Create the complete triplet model."""
    # Create embedding network
    embedding_net = create_embedding_network(input_shape, embedding_dim)
    
    # Define inputs
    anchor_input = layers.Input(shape=input_shape, name="anchor")
    positive_input = layers.Input(shape=input_shape, name="positive")
    negative_input = layers.Input(shape=input_shape, name="negative")
    
    # Generate embeddings
    anchor_embedding = embedding_net(anchor_input)
    positive_embedding = embedding_net(positive_input)
    negative_embedding = embedding_net(negative_input)
    
    # Stack embeddings for loss calculation
    embeddings = tf.stack([anchor_embedding, positive_embedding, negative_embedding], axis=1)
    
    # Create model
    model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=embeddings,
        name="TripletModel"
    )
    
    # Compile with custom loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=lambda y_true, y_pred: triplet_loss(y_true, y_pred, margin)
    )
    
    return model, embedding_net


def plot_training_history(history, metric: str = 'loss'):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric], label='Training')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'Model {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_triplets(anchors: np.ndarray, positives: np.ndarray, negatives: np.ndarray, num_samples: int = 3):
    """Visualize sample triplets."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle("Anchor | Positive | Negative", fontsize=16)
    
    for i in range(num_samples):
        # Denormalize images for display
        for j, img in enumerate([anchors[i], positives[i], negatives[i]]):
            if img.max() <= 1.0:  # If normalized
                display_img = (img * 255).astype(np.uint8)
            else:
                display_img = img.astype(np.uint8)
                
            axes[i, j].imshow(display_img)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluate_embeddings(embedding_model: Model, anchors: np.ndarray, positives: np.ndarray, 
                       negatives: np.ndarray, num_samples: int = 16):
    """Evaluate embedding quality on sample triplets."""
    # Generate embeddings
    anchor_emb = embedding_model.predict(anchors[:num_samples], verbose=0)
    positive_emb = embedding_model.predict(positives[:num_samples], verbose=0)
    negative_emb = embedding_model.predict(negatives[:num_samples], verbose=0)
    
    # Calculate distances
    pos_distances = np.sum(np.square(anchor_emb - positive_emb), axis=1)
    neg_distances = np.sum(np.square(anchor_emb - negative_emb), axis=1)
    
    # Calculate similarities
    pos_similarities = np.sum(anchor_emb * positive_emb, axis=1)
    neg_similarities = np.sum(anchor_emb * negative_emb, axis=1)
    
    print(f"\n--- Embedding Evaluation on {num_samples} samples ---")
    print(f"Average Positive Distance: {np.mean(pos_distances):.4f} ± {np.std(pos_distances):.4f}")
    print(f"Average Negative Distance: {np.mean(neg_distances):.4f} ± {np.std(neg_distances):.4f}")
    print(f"Average Positive Similarity: {np.mean(pos_similarities):.4f} ± {np.std(pos_similarities):.4f}")
    print(f"Average Negative Similarity: {np.mean(neg_similarities):.4f} ± {np.std(neg_similarities):.4f}")
    
    # Calculate accuracy
    correct_separations = np.sum(pos_distances < neg_distances)
    accuracy = correct_separations / num_samples
    print(f"Triplet Accuracy: {accuracy:.2%} ({correct_separations}/{num_samples})")
    
    return {
        'pos_dist': pos_distances,
        'neg_dist': neg_distances,
        'pos_sim': pos_similarities,
        'neg_sim': neg_similarities,
        'accuracy': accuracy
    }


def main():
    """Main training pipeline."""
    print("Starting Triplet Loss Face Recognition Training")
    print("=" * 50)
    
    # Load and prepare data
    print("\n1. Loading and preparing dataset...")
    anchors, positives, negatives = create_triplet_dataset(
        PATHS['image_dir'], 
        PATHS['pairs_file'], 
        CONFIG['target_shape']
    )
    
    # Shuffle and split data
    print("\n2. Shuffling and splitting dataset...")
    anchors, positives, negatives = shuffle_triplets(anchors, positives, negatives)
    train_anchors, train_positives, train_negatives, val_anchors, val_positives, val_negatives = split_dataset(
        anchors, positives, negatives, CONFIG['train_split']
    )
    
    print(f"Training samples: {len(train_anchors)}")
    print(f"Validation samples: {len(val_anchors)}")
    
    # Preprocess data
    print("\n3. Preprocessing images...")
    train_anchors, train_positives, train_negatives = preprocess_triplets(
        train_anchors, train_positives, train_negatives
    )
    val_anchors, val_positives, val_negatives = preprocess_triplets(
        val_anchors, val_positives, val_negatives
    )
    
    # Create model
    print("\n4. Creating triplet model...")
    input_shape = CONFIG['target_shape'] + (3,)
    triplet_model, embedding_model = create_triplet_model(
        input_shape, 
        CONFIG['embedding_dim'], 
        CONFIG['margin']
    )
    
    print(f"Model created with {triplet_model.count_params():,} parameters")
    
    # Train model
    print("\n5. Training model...")
    dummy_labels = np.zeros((len(train_anchors), 1))  # Dummy labels for triplet loss
    val_dummy_labels = np.zeros((len(val_anchors), 1))
    
    history = triplet_model.fit(
        x=[train_anchors, train_positives, train_negatives],
        y=dummy_labels,
        validation_data=([val_anchors, val_positives, val_negatives], val_dummy_labels),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        verbose=1
    )
    
    # Plot results
    print("\n6. Plotting training history...")
    plot_training_history(history)
    
    # Visualize sample triplets
    print("\n7. Visualizing sample triplets...")
    sample_indices = np.random.choice(len(val_anchors), 3, replace=False)
    sample_anchors = ((val_anchors[sample_indices] + 1) * 127.5).astype(np.uint8)
    sample_positives = ((val_positives[sample_indices] + 1) * 127.5).astype(np.uint8)
    sample_negatives = ((val_negatives[sample_indices] + 1) * 127.5).astype(np.uint8)
    
    visualize_triplets(sample_anchors, sample_positives, sample_negatives)
    
    # Evaluate embeddings
    print("\n8. Evaluating embeddings...")
    evaluate_embeddings(embedding_model, val_anchors, val_positives, val_negatives)
    
    print("\nTraining completed!")
    return triplet_model, embedding_model, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Run main pipeline
    model, embedding_net, training_history = main()