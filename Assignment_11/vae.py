import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model
import matplotlib.pyplot as plt

# Load the dataset
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
x_test = np.expand_dims(x_test.astype('float32') / 255., -1)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Configuration
LATENT_DIM = 2
IMAGE_SHAPE = (28, 28, 1)

def sampling(args):
    """Sampling function for VAE latent space"""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder():
    """Build encoder network"""
    encoder_inputs = Input(shape=IMAGE_SHAPE)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    
    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, encoder_inputs

def build_decoder():
    """Build decoder network"""
    decoder_inputs = Input(shape=(LATENT_DIM,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(decoder_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    
    decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
    return decoder, decoder_inputs

def vae_loss_function(loss_type='bce'):
    """Create VAE loss function"""
    def loss_fn(y_true, y_pred):
        # Get encoder and decoder from the model
        encoder = y_pred._keras_history[0].layer.layers[0]  # This is a hack, better to pass explicitly
        
        # Get latent variables
        z_mean, z_log_var, z = encoder(y_true)
        reconstruction = y_pred
        
        # Reconstruction loss
        if loss_type == 'bce':
            reconstruction_loss_tensor = keras.losses.binary_crossentropy(y_true, reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss_tensor, axis=(1, 2)))
        elif loss_type == 'mse':
            mse_loss_fn = keras.losses.MeanSquaredError(reduction="none")
            reconstruction_loss_tensor = mse_loss_fn(y_true, reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss_tensor, axis=(1, 2)))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # KL divergence loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = reconstruction_loss + kl_loss
        return total_loss
    
    return loss_fn

def build_vae_model(loss_type='bce'):
    """Build complete VAE model using functional approach"""
    # Build encoder and decoder
    encoder, encoder_inputs = build_encoder()
    decoder, decoder_inputs = build_decoder()
    
    # Connect encoder and decoder
    z_mean, z_log_var, z = encoder(encoder_inputs)
    decoder_outputs = decoder(z)
    
    # Create the full VAE model
    vae = Model(encoder_inputs, decoder_outputs, name=f'vae_{loss_type}')
    
    return vae, encoder, decoder

def train_vae_functional(x_train, x_test, loss_type='bce', epochs=50, batch_size=128):
    """Train VAE using functional approach with custom training loop"""
    print(f"\n--- Training VAE with {loss_type.upper()} Loss ---")
    
    # Build model
    vae, encoder, decoder = build_vae_model(loss_type)
    
    # Optimizer
    optimizer = keras.optimizers.Adam()
    
    # Metrics tracking
    train_loss_history = []
    train_recon_loss_history = []
    train_kl_loss_history = []
    val_loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        
        # Training
        epoch_loss = []
        epoch_recon_loss = []
        epoch_kl_loss = []
        
        # Batch training
        num_batches = len(x_train) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = x_train[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                # Forward pass
                z_mean, z_log_var, z = encoder(batch_data)
                reconstruction = decoder(z)
                
                # Calculate losses
                if loss_type == 'bce':
                    reconstruction_loss_tensor = keras.losses.binary_crossentropy(batch_data, reconstruction)
                    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss_tensor, axis=(1, 2)))
                elif loss_type == 'mse':
                    mse_loss_fn = keras.losses.MeanSquaredError(reduction="none")
                    reconstruction_loss_tensor = mse_loss_fn(batch_data, reconstruction)
                    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss_tensor, axis=(1, 2)))
                
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                
                total_loss = reconstruction_loss + kl_loss
            
            # Backward pass
            gradients = tape.gradient(total_loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(gradients, vae.trainable_weights))
            
            epoch_loss.append(total_loss.numpy())
            epoch_recon_loss.append(reconstruction_loss.numpy())
            epoch_kl_loss.append(kl_loss.numpy())
        
        # Calculate epoch averages
        avg_train_loss = np.mean(epoch_loss)
        avg_recon_loss = np.mean(epoch_recon_loss)
        avg_kl_loss = np.mean(epoch_kl_loss)
        
        train_loss_history.append(avg_train_loss)
        train_recon_loss_history.append(avg_recon_loss)
        train_kl_loss_history.append(avg_kl_loss)
        
        # Validation
        z_mean_val, z_log_var_val, z_val = encoder(x_test)
        reconstruction_val = decoder(z_val)
        
        if loss_type == 'bce':
            val_recon_loss_tensor = keras.losses.binary_crossentropy(x_test, reconstruction_val)
            val_recon_loss = tf.reduce_mean(tf.reduce_sum(val_recon_loss_tensor, axis=(1, 2)))
        elif loss_type == 'mse':
            mse_loss_fn = keras.losses.MeanSquaredError(reduction="none")
            val_recon_loss_tensor = mse_loss_fn(x_test, reconstruction_val)
            val_recon_loss = tf.reduce_mean(tf.reduce_sum(val_recon_loss_tensor, axis=(1, 2)))
        
        val_kl_loss = -0.5 * (1 + z_log_var_val - tf.square(z_mean_val) - tf.exp(z_log_var_val))
        val_kl_loss = tf.reduce_mean(tf.reduce_sum(val_kl_loss, axis=1))
        val_total_loss = val_recon_loss + val_kl_loss
        
        val_loss_history.append(val_total_loss.numpy())
        
        print(f'Loss: {avg_train_loss:.4f} - Recon: {avg_recon_loss:.4f} - KL: {avg_kl_loss:.4f} - Val Loss: {val_total_loss:.4f}')
    
    # Create history dictionary for compatibility
    history = {
        'total_loss': train_loss_history,
        'reconstruction_loss': train_recon_loss_history,
        'kl_loss': train_kl_loss_history,
        'val_total_loss': val_loss_history
    }
    
    return vae, encoder, decoder, history

# Train both models
print("Building and training models...")

# Train BCE model
vae_bce, encoder_bce, decoder_bce, history_bce = train_vae_functional(
    x_train, x_test, loss_type='bce', epochs=50, batch_size=128
)

# Train MSE model  
vae_mse, encoder_mse, decoder_mse, history_mse = train_vae_functional(
    x_train, x_test, loss_type='mse', epochs=50, batch_size=64
)

print("\nTraining completed!")

# Visualization and comparison
n = 8
sample_images = x_test[:n]

# Generate reconstructions
reconstructed_bce = vae_bce.predict(sample_images)
reconstructed_mse = vae_mse.predict(sample_images)

# Generate latent representations
z_mean_bce, _, _ = encoder_bce.predict(x_test, batch_size=128)
z_mean_mse, _, _ = encoder_mse.predict(x_test, batch_size=128)

# Plot reconstructions
plt.figure(figsize=(15, 6))
plt.suptitle("VAE Reconstruction Comparison: BCE vs. MSE", fontsize=16)

for i in range(n):
    # Original Image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    ax.set_title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # BCE Reconstructed Image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_bce[i].reshape(28, 28), cmap='gray')
    ax.set_title("BCE Recon")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # MSE Reconstructed Image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(reconstructed_mse[i].reshape(28, 28), cmap='gray')
    ax.set_title("MSE Recon")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot latent spaces
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("VAE Latent Space Comparison: BCE vs. MSE", fontsize=16)

# Latent Space from BCE-trained VAE
scatter1 = ax1.scatter(z_mean_bce[:, 0], z_mean_bce[:, 1], c=y_test, cmap='viridis', s=5, alpha=0.7)
ax1.set_xlabel("Latent Dimension 1")
ax1.set_ylabel("Latent Dimension 2")
ax1.set_title("Latent Space (BCE Loss)")
ax1.grid(True)
fig.colorbar(scatter1, ax=ax1, label='Digit Class')

# Latent Space from MSE-trained VAE
scatter2 = ax2.scatter(z_mean_mse[:, 0], z_mean_mse[:, 1], c=y_test, cmap='viridis', s=5, alpha=0.7)
ax2.set_xlabel("Latent Dimension 1")
ax2.set_ylabel("Latent Dimension 2")
ax2.set_title("Latent Space (MSE Loss)")
ax2.grid(True)
fig.colorbar(scatter2, ax=ax2, label='Digit Class')

plt.show()

# Plot training history comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history_bce['total_loss'], label='BCE Train')
plt.plot(history_bce['val_total_loss'], label='BCE Val')
plt.plot(history_mse['total_loss'], label='MSE Train')
plt.plot(history_mse['val_total_loss'], label='MSE Val')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history_bce['reconstruction_loss'], label='BCE')
plt.plot(history_mse['reconstruction_loss'], label='MSE')
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history_bce['kl_loss'], label='BCE')
plt.plot(history_mse['kl_loss'], label='MSE')
plt.title('KL Divergence Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nComparison completed! Check the plots to see the differences between BCE and MSE loss functions.")


