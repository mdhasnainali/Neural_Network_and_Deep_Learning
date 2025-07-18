import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# --- Constants ---
LATENT_DIM = 2
IMAGE_SHAPE = (28, 28, 1)


# ====================
# 1. Data Preparation
# ====================
def load_mnist_data():
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
    x_test = np.expand_dims(x_test.astype('float32') / 255., -1)
    return x_train, x_test, y_test


# ====================
# 2. VAE Components
# ====================
def build_encoder():
    encoder_inputs = layers.Input(shape=IMAGE_SHAPE)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)

    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder


def build_decoder():
    decoder_inputs = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(decoder_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
    return decoder


# ====================
# 3. VAE Class
# ====================
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
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def _calculate_loss(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss)

        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self._calculate_loss(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        total_loss, reconstruction_loss, kl_loss = self._calculate_loss(data)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# ====================
# 4. Evaluation Utilities
# ====================
def plot_reconstruction_quality(model, x_test, n=5):
    sample_images = x_test[:n]
    _, _, z_samples = model.encoder.predict(sample_images)
    reconstructed_images = model.decoder.predict(z_samples)

    plt.figure(figsize=(15, 4))
    plt.suptitle("VAE Reconstruction Quality", fontsize=16)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_latent_space(model, x_test, y_test):
    z_mean, _, _ = model.encoder.predict(x_test, batch_size=128)
    plt.figure(figsize=(10, 8))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='viridis', s=5)
    plt.colorbar()
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("2D Latent Space of MNIST")
    plt.grid(True)
    plt.show()


# ====================
# 5. Image Generation from Noise
# ====================
def generate_from_noise(decoder, n=5, mean=5.0, variance=1.0):
    noise_stddev = np.sqrt(variance)
    noise_vectors = np.random.normal(loc=mean, scale=noise_stddev, size=(n, LATENT_DIM))
    generated_images = decoder.predict(noise_vectors)

    plt.figure(figsize=(15, 6))
    plt.suptitle("Denoising Decoder: Noise Vectors and their Generated Images", fontsize=16)

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(noise_vectors[i].reshape(LATENT_DIM, 1), cmap='viridis')
        ax.set_title(f"Noise {i+1}")
        ax.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        ax.set_title("Generated")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ====================
# 6. Main Pipeline
# ====================
def main():
    x_train, x_test, y_test = load_mnist_data()
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    print("\n--- Training VAE ---")
    vae.fit(x_train, epochs=20, batch_size=128, validation_data=(x_test, None))

    print("\n--- Evaluating Reconstruction ---")
    plot_reconstruction_quality(vae, x_test, n=5)

    print("\n--- Visualizing Latent Space ---")
    plot_latent_space(vae, x_test, y_test)

    print("\n--- Generating Images from Noise ---")
    generate_from_noise(decoder, n=5, mean=5.0, variance=1.0)


if __name__ == "__main__":
    main()
