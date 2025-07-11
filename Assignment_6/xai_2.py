import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_model():
    model = tf.keras.applications.InceptionResNetV2(include_top=True, weights='imagenet')
    model.trainable = False
    return model

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    return tf.expand_dims(image, 0)

def get_prediction(model, image):
    predictions = model.predict(image, verbose=0)
    decoded = tf.keras.applications.inception_resnet_v2.decode_predictions(predictions, top=1)
    class_name = decoded[0][0][1]
    confidence = decoded[0][0][2]
    class_idx = np.argmax(predictions[0])
    return class_name, confidence, class_idx, predictions 

def create_logits_model(model):
    # Get the layer before the final predictions layer
    avg_pool_output = model.get_layer('avg_pool').output
    
    # Get the weights and bias from the predictions layer
    predictions_layer = model.get_layer('predictions')
    weights, bias = predictions_layer.get_weights()
    
    # Create logits by applying dense transformation without softmax
    logits = tf.keras.layers.Dense(
        units=weights.shape[1],
        use_bias=True,
        activation=None,  # No activation (no softmax)
        name='logits'
    )(avg_pool_output)
    
    # Set the weights to match the original predictions layer
    logits_model = tf.keras.Model(inputs=model.input, outputs=logits)
    logits_model.get_layer('logits').set_weights([weights, bias])
    
    return logits_model

def generate_gradcam_softmax(model, image, class_idx):
    last_conv_layer = model.get_layer('conv_7b')
    
    # Create a model that outputs both conv features and final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        # Use softmax probability (not logits) for gradient computation
        class_output = predictions[:, class_idx]
    
    # Compute gradients of softmax output conv features
    grads = tape.gradient(class_output, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by the gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_standard(model, image, class_idx):
    last_conv_layer = model.get_layer('conv_7b')
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_output = predictions[:, class_idx]
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_integrated_gradients_logits(model, image, class_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = tf.zeros_like(image)
    
    logits_model = create_logits_model(model)
    
    # Generate interpolated images
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated_images = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        interpolated_images.append(interpolated)
    
    interpolated_images = tf.concat(interpolated_images, axis=0)
    
    # Compute gradients logits
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        logits = logits_model(interpolated_images)
        target_class_logits = logits[:, class_idx]
    
    gradients = tape.gradient(target_class_logits, interpolated_images)
    
    # Average gradients and compute integrated gradients
    avg_gradients = tf.reduce_mean(gradients, axis=0, keepdims=True)
    integrated_gradients = (image - baseline) * avg_gradients
    
    # Sum across color channels to get attribution map
    attribution = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)
    attribution = attribution[0]
    
    return attribution.numpy()

def generate_integrated_gradients_standard(model, image, class_idx, baseline=None, steps=50):
    """
    Standard Integrated Gradients using softmax probabilities.
    """
    if baseline is None:
        baseline = tf.zeros_like(image)
    
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated_images = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        interpolated_images.append(interpolated)
    
    interpolated_images = tf.concat(interpolated_images, axis=0)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images)
        target_class_probs = predictions[:, class_idx]
    
    gradients = tape.gradient(target_class_probs, interpolated_images)
    
    avg_gradients = tf.reduce_mean(gradients, axis=0, keepdims=True)
    integrated_gradients = (image - baseline) * avg_gradients
    
    attribution = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)
    attribution = attribution[0]
    
    return attribution.numpy()

def create_overlay(image, heatmap, alpha=0.4):
    """Create overlay of heatmap on image"""
    img = image[0].numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
    
    heatmap_resized = tf.image.resize(
        tf.expand_dims(heatmap, -1),
        (299, 299)
    ).numpy()[:, :, 0]
    
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    overlay = heatmap_colored * alpha + img * (1 - alpha)
    return overlay

def analyze_attribution(image_path):

    model = load_model()
    
    original_image = preprocess_image(image_path)
    
    orig_class, orig_conf, orig_idx, _ = get_prediction(model, original_image)
    print(f"Original prediction: {orig_class} ({orig_conf:.2%} confidence)")
    
    gradcam_softmax = generate_gradcam_softmax(model, original_image, orig_idx)
    
    ig_logits = generate_integrated_gradients_logits(model, original_image, orig_idx)
    
    gradcam_standard = generate_gradcam_standard(model, original_image, orig_idx)
    
    ig_standard = generate_integrated_gradients_standard(model, original_image, orig_idx)
    
    create_attribution_comparison_plot(
        original_image,
        gradcam_softmax,
        ig_logits,
        gradcam_standard,
        ig_standard,
        orig_class,
        orig_conf
    )
    

def create_attribution_comparison_plot(orig_img, gradcam_softmax, ig_logits, gradcam_standard, ig_standard, orig_class, orig_conf):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    gradcam_softmax_overlay = create_overlay(orig_img, gradcam_softmax)
    axes[0, 0].imshow(gradcam_softmax_overlay)
    axes[0, 0].set_title('Grad-CAM\n(Gradients from Softmax)', 
                        fontsize=12, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ig_logits, cmap='hot')
    axes[0, 1].set_title('Integrated Gradients\n(Gradients from Logits)', 
                        fontsize=12, fontweight='bold', color='red')
    axes[0, 1].axis('off')

    gradcam_standard_overlay = create_overlay(orig_img, gradcam_standard)
    axes[1, 0].imshow(gradcam_standard_overlay)
    axes[1, 0].set_title('Standard Grad-CAM\n(Conv Layer Features)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(ig_standard, cmap='hot')
    axes[1, 1].set_title('Standard Integrated Gradients\n(Softmax Probabilities)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.suptitle(f"Predicted: {orig_class} (Confidence: {orig_conf:.1%})",
                 fontsize=14, fontweight='bold')
    
    output_dir = "xai_softmax_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = "xai_softmax_logits_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    image_path = "./plane.jpeg" 
    
    analyze_attribution(image_path=image_path)
