import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

# ----------------------- #
# Configuration
# ----------------------- #
LATENT_DIM = 2
IMAGE_SHAPE = (28, 28, 1)
EPOCHS = 20
BATCH_SIZE = 128


# ----------------------- #
# Data Preparation
# ----------------------- #
def load_mnist_data():
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
    x_test = np.expand_dims(x_test.astype('float32') / 255., -1)
    return x_train, x_test, y_test


# ----------------------- #
# VAE Sampling Layer
# ----------------------- #
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ----------------------- #
# Encoder Definition
# ----------------------- #
def build_encoder(latent_dim):
    inputs = Input(shape=IMAGE_SHAPE)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z], name='encoder')


# ----------------------- #
# Decoder Definition
# ----------------------- #
def build_decoder(latent_dim):
    inputs = Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return Model(inputs, outputs, name='decoder')


# ----------------------- #
# Custom VAE Model Class
# ----------------------- #
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def compute_loss(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(inputs, reconstruction), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        total_loss, reconstruction_loss, kl_loss = self.compute_loss(data)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# ----------------------- #
# Visualization Utilities
# ----------------------- #
def visualize_reconstruction(vae, x_test, n=5):
    sample_images = x_test[n:n+5]
    _, _, z_samples = vae.encoder.predict(sample_images)
    reconstructed_images = vae.decoder.predict(z_samples)

    plt.figure(figsize=(15, 4))
    plt.suptitle("VAE Reconstruction Quality", fontsize=16)
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i].squeeze(), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off')
    plt.show()


def visualize_latent_space(encoder, x_test, y_test):
    z_mean, _, _ = encoder.predict(x_test, batch_size=128)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(label='Digit Label')
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("2D Latent Space Visualization (VAE)", fontsize=16)
    plt.grid(True)
    plt.show()


# ----------------------- #
# Main Execution
# ----------------------- #
def main():
    # Load data
    x_train, x_test, y_test = load_mnist_data()
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # Build model
    encoder = build_encoder(LATENT_DIM)
    decoder = build_decoder(LATENT_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    print("\n--- Training the VAE ---")
    vae.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, None))

    # Visualization
    visualize_reconstruction(vae, x_test)
    visualize_latent_space(encoder, x_test, y_test)


if __name__ == '__main__':
    main()
