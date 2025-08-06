import keras
import keras.ops as ops
import matplotlib.pyplot as plt
import numpy as np


def euclidean_distance(vects):
    """Calculate euclidean distance between two vectors."""
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


def create_embedding_network(input_shape):
    """Create embedding network using ResNet50V2 backbone."""
    # Define the ResNet50V2 base model
    base_cnn = keras.applications.ResNet50V2(
        weights="imagenet", 
        input_shape=input_shape,
        include_top=False,
    )

    # Set layers to be trainable after conv5_block1_out
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    # Add custom embedding head
    flatten = keras.layers.Flatten()(base_cnn.output)
    dense1 = keras.layers.Dense(512, activation="relu")(flatten)
    dense1 = keras.layers.BatchNormalization()(dense1)
    embedding = keras.layers.Dense(256)(dense1)  # 256-dimensional embedding

    return keras.Model(base_cnn.input, embedding, name="EmbeddingNetwork")


def create_siamese_model(input_shape):
    """Create complete Siamese network model."""
    # Create embedding network
    embedding_network = create_embedding_network(input_shape)
    
    # Define inputs for Siamese network
    input_1 = keras.layers.Input(input_shape, name="input_1")
    input_2 = keras.layers.Input(input_shape, name="input_2")
    
    # Generate embeddings
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)
    
    # Calculate distance and make prediction
    distance = keras.layers.Lambda(
        euclidean_distance, 
        output_shape=(1,),
        name="euclidean_distance"
    )([tower_1, tower_2])
    
    normalized_distance = keras.layers.BatchNormalization(name="distance_norm")(distance)
    prediction = keras.layers.Dense(1, activation="sigmoid", name="similarity_output")(normalized_distance)
    
    # Create and compile model
    model = keras.Model(inputs=[input_1, input_2], outputs=prediction, name="SiameseNetwork")
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )
    
    return model, embedding_network


def train_siamese_model(model, train_data, val_data, batch_size=32, epochs=20):
    """Train the Siamese model."""
    x_train_1, x_train_2, labels_train = train_data
    x_val_1, x_val_2, labels_val = val_data
    
    history = model.fit(
        [x_train_1, x_train_2],
        labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    return history


def plot_training_metrics(history, save_path=None):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history["accuracy"], label="Training")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history["loss"], label="Training")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Binary Cross-Entropy Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_data, batch_size=32):
    """Evaluate model on test data."""
    x_test_1, x_test_2, labels_test = test_data
    
    results = model.evaluate([x_test_1, x_test_2], labels_test, batch_size=batch_size, verbose=1)
    print(f"\nTest Results - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
    
    # Generate predictions
    predictions = model.predict([x_test_1, x_test_2], batch_size=batch_size)
    
    return results, predictions


def visualize_predictions(pairs, labels, predictions, num_samples=6, num_cols=3, figsize=(12, 8)):
    """Visualize test pairs with predictions."""
    num_rows = (num_samples + num_cols - 1) // num_cols
    num_samples = min(num_samples, len(pairs))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        
        # Denormalize images for display (assuming ResNet preprocessing)
        img_pair = (pairs[i] / 2.0) + 0.5
        img_pair = np.clip(img_pair, 0, 1)
        
        # Concatenate images side by side
        combined_img = np.concatenate([img_pair[0], img_pair[1]], axis=1)
        ax.imshow(combined_img)
        ax.set_axis_off()
        
        # Set title with true label and prediction
        true_label = int(labels[i])
        pred_prob = predictions[i][0]
        pred_label = "Same" if pred_prob > 0.5 else "Different"
        
        ax.set_title(f"True: {'Same' if true_label == 1 else 'Different'} | "
                    f"Pred: {pred_label} ({pred_prob:.3f})", 
                    fontsize=10)
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main_pipeline(train_data, val_data, test_data, pairs_test, 
                  input_shape, batch_size=32, epochs=20):
    """Complete training and evaluation pipeline."""
    print("Creating Siamese network...")
    siamese_model, embedding_network = create_siamese_model(input_shape)
    siamese_model.summary()
    
    print(f"\nTraining for {epochs} epochs...")
    history = train_siamese_model(siamese_model, train_data, val_data, batch_size, epochs)
    
    print("\nPlotting training metrics...")
    plot_training_metrics(history)
    
    print("\nEvaluating on test set...")
    test_results, predictions = evaluate_model(siamese_model, test_data, batch_size)
    
    print("\nVisualizing predictions...")
    visualize_predictions(pairs_test, test_data[2], predictions, num_samples=6)
    
    return siamese_model, embedding_network, history, test_results, predictions



# Define your data and parameters
TARGET_SHAPE = (224, 224)  # or whatever your target shape is
input_shape = TARGET_SHAPE + (3,)

# Prepare your data as tuples
train_data = (x_train_1, x_train_2, labels_train)
val_data = (x_val_1, x_val_2, labels_val)
test_data = (x_test_1, x_test_2, labels_test)

# Run the complete pipeline
model, embedding_net, history, results, preds = main_pipeline(
    train_data, val_data, test_data, pairs_test,
    input_shape, batch_size=32, epochs=20
)
