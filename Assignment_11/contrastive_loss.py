import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import ops
from keras.applications.resnet_v2 import preprocess_input

# ----------------------------#
# Configuration
# ----------------------------#
LFW_DIR = "LFW/"
IMAGE_DIR = os.path.join(LFW_DIR, "lfw-deepfunneled")
PAIRS_FILE = os.path.join(LFW_DIR, "pairs.csv")
TARGET_SHAPE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
MARGIN = 1


# ----------------------------#
# Data Loading & Processing
# ----------------------------#
def load_and_create_pairs_from_lfw(image_dir, pairs_path, target_shape):
    """
    Loads positive and negative face image pairs from LFW and returns them as arrays.

    Returns:
        pairs (np.ndarray): Array of image pairs.
        labels (np.ndarray): Array of 0 (same person) or 1 (different people).
    """
    pairs, labels = [], []
    positive_paths, negative_paths = [], []

    with open(pairs_path, 'r') as f:
        next(f)  # Skip header
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) in {3, 4} and (len(parts) == 3 or parts[3] == ''):
                p1, n1, n2 = parts[0], int(parts[1]), int(parts[2])
                path1 = os.path.join(image_dir, p1, f"{p1}_{n1:04d}.jpg")
                path2 = os.path.join(image_dir, p1, f"{p1}_{n2:04d}.jpg")
                positive_paths.append([path1, path2])
            elif len(parts) == 4:
                p1, n1, p2, n2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                path1 = os.path.join(image_dir, p1, f"{p1}_{n1:04d}.jpg")
                path2 = os.path.join(image_dir, p2, f"{p2}_{n2:04d}.jpg")
                negative_paths.append([path1, path2])

    all_path_pairs = positive_paths + negative_paths
    all_labels = [0] * len(positive_paths) + [1] * len(negative_paths)

    for i, (path1, path2) in enumerate(all_path_pairs):
        try:
            img1 = Image.open(path1).convert("RGB").resize(target_shape)
            img2 = Image.open(path2).convert("RGB").resize(target_shape)
            pairs.append([np.array(img1, dtype="float32"), np.array(img2, dtype="float32")])
            labels.append(all_labels[i])
        except FileNotFoundError:
            print(f"Warning: Could not load pair {i}: {path1}, {path2}")
            continue

    return np.array(pairs), np.array(labels, dtype="float32")


# ----------------------------#
# Model Definition
# ----------------------------#
def euclidean_distance(vects):
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


def create_embedding_network(input_shape):
    base_cnn = keras.applications.ResNet50V2(weights="imagenet", input_shape=input_shape, include_top=False)
    
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    flatten = keras.layers.Flatten()(base_cnn.output)
    dense1 = keras.layers.Dense(512, activation="relu")(flatten)
    dense1 = keras.layers.BatchNormalization()(dense1)
    output = keras.layers.Dense(256)(dense1)
    return keras.Model(base_cnn.input, output, name="Embedding")


def build_siamese_network(input_shape, margin):
    embedding_network = create_embedding_network(input_shape)
    input_1 = keras.layers.Input(input_shape)
    input_2 = keras.layers.Input(input_shape)

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)
    distance = keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
    norm = keras.layers.BatchNormalization()(distance)
    output = keras.layers.Dense(1, activation="sigmoid")(norm)

    model = keras.Model(inputs=[input_1, input_2], outputs=output)
    
    def contrastive_loss(y_true, y_pred):
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - y_pred, 0))
        return ops.mean((1 - y_true) * square_pred + y_true * margin_square)

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=contrastive_loss, metrics=["accuracy"])
    return model


# ----------------------------#
# Training Utilities
# ----------------------------#
def split_and_preprocess(pairs, labels):
    p = np.random.permutation(len(pairs))
    pairs, labels = pairs[p], labels[p]

    # 80% train+val, 20% test
    train_val_idx = int(0.8 * len(pairs))
    val_idx = int(0.5 * train_val_idx)

    pairs_train_val, labels_train_val = pairs[:train_val_idx], labels[:train_val_idx]
    pairs_test, labels_test = pairs[train_val_idx:], labels[train_val_idx:]

    pairs_train, labels_train = pairs_train_val[:val_idx], labels_train_val[:val_idx]
    pairs_val, labels_val = pairs_train_val[val_idx:], labels_train_val[val_idx:]

    def preprocess_pairs(pairs):
        return preprocess_input(pairs[:, 0]), preprocess_input(pairs[:, 1])

    return (
        preprocess_pairs(pairs_train) + (labels_train,),
        preprocess_pairs(pairs_val) + (labels_val,),
        preprocess_pairs(pairs_test) + (labels_test,)
    )


def plot_metric(history, metric, title, filename=None):
    plt.figure()
    plt.plot(history.history[metric])
    if "val_" + metric in history.history:
        plt.plot(history.history["val_" + metric])
        plt.legend(["Train", "Validation"])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    if filename:
        plt.savefig(filename)
    plt.show()


def visualize_predictions(pairs, labels, predictions=None, to_show=6, num_col=3, test=False, filename=None):
    num_row = (to_show + num_col - 1) // num_col
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    axes = axes.flatten()

    for i in range(to_show):
        img_pair = (pairs[i] / 255.0)  # Normalize for plotting
        ax = axes[i]
        ax.imshow(np.concatenate([img_pair[0], img_pair[1]], axis=1))
        ax.set_axis_off()
        if test:
            ax.set_title(f"True: {int(labels[i])} | Pred: {predictions[i][0]:.2f}")
        else:
            ax.set_title(f"Label: {int(labels[i])}")

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


# ----------------------------#
# Execution
# ----------------------------#
if __name__ == "__main__":
    all_pairs, all_labels = load_and_create_pairs_from_lfw(IMAGE_DIR, PAIRS_FILE, TARGET_SHAPE)
    print(f"\nTotal pairs loaded: {len(all_pairs)}")

    (x_train_1, x_train_2, y_train), (x_val_1, x_val_2, y_val), (x_test_1, x_test_2, y_test) = split_and_preprocess(all_pairs, all_labels)

    siamese_model = build_siamese_network(TARGET_SHAPE + (3,), margin=MARGIN)
    siamese_model.summary()

    history = siamese_model.fit(
        [x_train_1, x_train_2],
        y_train,
        validation_data=([x_val_1, x_val_2], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Save plots
    plot_metric(history, "accuracy", "Model Accuracy", "accuracy_plot.png")
    plot_metric(history, "loss", "Contrastive Loss", "loss_plot.png")

    # Evaluation
    results = siamese_model.evaluate([x_test_1, x_test_2], y_test, batch_size=BATCH_SIZE)
    print("\nTest Loss, Test Accuracy:", results)

    # Visualization
    predictions = siamese_model.predict([x_test_1, x_test_2])
    visualize_predictions(all_pairs[len(all_pairs) - len(x_test_1):], y_test, predictions=predictions, to_show=6, test=True, filename="pred.png")
