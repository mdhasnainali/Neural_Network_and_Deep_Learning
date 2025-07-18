import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# ----------------------------- #
#       Configurations          #
# ----------------------------- #
LATENT_DIM = 32
EPOCHS = 25
BATCH_SIZE = 128
NOISE_FACTOR = 0.4
NOISE_MEAN = 5.0
NOISE_STDDEV = 1.0
NUM_SAMPLES = 5

# ----------------------------- #
#        Data Pipeline          #
# ----------------------------- #
def load_and_prepare_data(noise_factor=0.4):
    (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, -1)  # (28, 28, 1)
    x_test = np.expand_dims(x_test, -1)

    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train, x_test, x_train_noisy, x_test_noisy

def visualize_noisy_data(x_clean, x_noisy, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_clean[i].squeeze(), cmap='gray')
        plt.title("Original")
        ax.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_noisy[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        ax.axis('off')
    plt.show()

# ----------------------------- #
#      Model Architectures      #
# ----------------------------- #
def build_encoder(latent_dim):
    encoder_input = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)  # 14x14
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)              # 7x7
    shape_before_flattening = x.shape[1:]
    x = layers.Flatten()(x)
    encoder_output = layers.Dense(latent_dim, activation='relu')(x)
    encoder = models.Model(encoder_input, encoder_output, name='encoder')
    return encoder, shape_before_flattening

def build_decoder(latent_dim, shape_before_flattening):
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(shape_before_flattening), activation='relu')(decoder_input)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    return decoder

def build_denoising_autoencoder(encoder, decoder):
    autoencoder_input = layers.Input(shape=(28, 28, 1))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name='denoising_autoencoder')
    return autoencoder

# ----------------------------- #
#         Training Code         #
# ----------------------------- #
def train_autoencoder(model, x_train_noisy, x_train, x_test_noisy, x_test):
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(
        x_train_noisy, x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test_noisy, x_test)
    )
    return history

# ----------------------------- #
#       Latent Sampling         #
# ----------------------------- #
def generate_noise_vectors(n, latent_dim, mean=0.0, stddev=1.0):
    return np.random.normal(loc=mean, scale=stddev, size=(n, latent_dim))

def visualize_generated_images(noise_vectors, decoder, title="Generated from Decoder"):
    generated_images = decoder.predict(noise_vectors)

    plt.figure(figsize=(15, 6))
    plt.suptitle(title, fontsize=16)

    for i in range(len(noise_vectors)):
        # Noise vector
        ax = plt.subplot(2, len(noise_vectors), i + 1)
        plt.imshow(noise_vectors[i].reshape(8, 4), cmap='viridis')
        plt.title(f"Noise Vector {i+1}")
        ax.axis('off')

        # Generated image
        ax = plt.subplot(2, len(noise_vectors), i + 1 + len(noise_vectors))
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.title("Generated Image")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ----------------------------- #
#             Main              #
# ----------------------------- #
def main():
    # Load data
    x_train, x_test, x_train_noisy, x_test_noisy = load_and_prepare_data(NOISE_FACTOR)

    print('x_train shape:', x_train.shape)
    print(f"{x_train.shape[0]} train samples")

    # Visualize samples
    visualize_noisy_data(x_test, x_test_noisy)

    # Build models
    encoder, shape_before_flattening = build_encoder(LATENT_DIM)
    decoder = build_decoder(LATENT_DIM, shape_before_flattening)
    autoencoder = build_denoising_autoencoder(encoder, decoder)

    print("\n--- Encoder Summary ---")
    encoder.summary()
    print("\n--- Decoder Summary ---")
    decoder.summary()
    print("\n--- Denoising Autoencoder Summary ---")
    autoencoder.summary()

    # Train
    print("\n--- Training the Denoising Autoencoder ---")
    train_autoencoder(autoencoder, x_train_noisy, x_train, x_test_noisy, x_test)

    # Generate and visualize
    noise_vectors = generate_noise_vectors(NUM_SAMPLES, LATENT_DIM, mean=NOISE_MEAN, stddev=NOISE_STDDEV)
    print(f"Shape of generated noise vectors: {noise_vectors.shape}")
    visualize_generated_images(noise_vectors, decoder, title="Denoising Decoder: Noise Vectors and Generated Images")

if __name__ == '__main__':
    main()
