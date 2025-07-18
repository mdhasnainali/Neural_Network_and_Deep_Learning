import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.datasets import mnist

# ---------------------------- #
#         Configs             #
# ---------------------------- #
LATENT_DIM = 32
INPUT_SHAPE = (28, 28)
INPUT_DIM = 784
EPOCHS = 20
BATCH_SIZE = 256
NOISE_MEAN = 5.0
NOISE_STDDEV = 1.0
NUM_SAMPLES = 5

# ---------------------------- #
#      Data Preparation       #
# ---------------------------- #
def load_and_preprocess_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train_flat = x_train.reshape((len(x_train), INPUT_DIM))
    x_test_flat = x_test.reshape((len(x_test), INPUT_DIM))

    return x_train_flat, x_test_flat

# ---------------------------- #
#     Model Architectures     #
# ---------------------------- #
def build_encoder(input_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    outputs = layers.Dense(latent_dim, activation='relu')(x)
    return models.Model(inputs, outputs, name='encoder')

def build_decoder(latent_dim, output_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    outputs = layers.Dense(output_dim, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name='decoder')

def build_autoencoder(encoder, decoder):
    inputs = layers.Input(shape=(INPUT_DIM,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    return models.Model(inputs, decoded, name='autoencoder')

# ---------------------------- #
#       Training Process      #
# ---------------------------- #
def train_autoencoder(autoencoder, x_train, x_test):
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    history = autoencoder.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test, x_test)
    )
    return history

# ---------------------------- #
#   Noise Generation & Plot   #
# ---------------------------- #
def generate_noise_vectors(n, latent_dim, mean=0.0, stddev=1.0):
    return np.random.normal(loc=mean, scale=stddev, size=(n, latent_dim))

def visualize_generated_images(noise_vectors, decoder, title="Generated Images"):
    generated = decoder.predict(noise_vectors)
    images = generated.reshape((-1, 28, 28))

    plt.figure(figsize=(15, 6))
    plt.suptitle("Noise Vectors and their Generated Images", fontsize=16)

    for i in range(len(noise_vectors)):
        # Plot noise vector
        ax = plt.subplot(2, len(noise_vectors), i + 1)
        plt.imshow(noise_vectors[i].reshape(8, 4), cmap='viridis')
        plt.title(f"Noise {i+1}")
        ax.axis('off')

        # Plot generated image
        ax = plt.subplot(2, len(noise_vectors), i + 1 + len(noise_vectors))
        plt.imshow(images[i], cmap='gray')
        plt.title("Generated Image")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ---------------------------- #
#            Main             #
# ---------------------------- #
def main():
    # Prepare data
    x_train, x_test = load_and_preprocess_data()
    print('Training Data Shape:', x_train.shape)
    print('Test Data Shape:', x_test.shape)

    # Build models
    encoder = build_encoder(INPUT_DIM, LATENT_DIM)
    decoder = build_decoder(LATENT_DIM, INPUT_DIM)
    autoencoder = build_autoencoder(encoder, decoder)

    print("\n--- Encoder Summary ---")
    encoder.summary()
    print("\n--- Decoder Summary ---")
    decoder.summary()
    print("\n--- Autoencoder Summary ---")
    autoencoder.summary()

    # Train
    print("\n--- Training Autoencoder ---")
    train_autoencoder(autoencoder, x_train, x_test)

    # Generate from noise
    noise_vectors = generate_noise_vectors(NUM_SAMPLES, LATENT_DIM, mean=NOISE_MEAN, stddev=NOISE_STDDEV)
    print("Noise Vectors Shape:", noise_vectors.shape)
    visualize_generated_images(noise_vectors, decoder)

if __name__ == '__main__':
    main()
