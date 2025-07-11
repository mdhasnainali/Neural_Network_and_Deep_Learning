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
    return class_name, confidence, class_idx

def create_adversarial_example(model, image, target_class=207, epsilon=0.2):
    """Create adversarial example using FGSM"""
    # Create target label (one-hot encoded)
    target_label = tf.one_hot(target_class, 1000)
    target_label = tf.reshape(target_label, (1, 1000))
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = tf.keras.losses.categorical_crossentropy(target_label, predictions)
    
    # Get gradients and create adversarial
    gradients = tape.gradient(loss, image)
    signed_gradients = tf.sign(gradients)
    adversarial_image = image + epsilon * signed_gradients
    adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)
    
    return adversarial_image

def generate_gradcam(model, image, class_idx):
    """Generate Grad-CAM heatmap"""
    # Create model that outputs both conv layer and predictions
    last_conv_layer = model.get_layer('conv_7b')
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_output = predictions[:, class_idx]
    
    # Calculate gradients
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_integrated_gradients(model, image, class_idx, baseline=None, steps=50):
    """Generate Integrated Gradients attribution"""
    if baseline is None:
        baseline = tf.zeros_like(image)
    
    # Generate interpolated images
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated_images = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (image - baseline)
        interpolated_images.append(interpolated)
    
    interpolated_images = tf.concat(interpolated_images, axis=0)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = model(interpolated_images)
        target_class_logits = predictions[:, class_idx]
    
    gradients = tape.gradient(target_class_logits, interpolated_images)
    
    avg_gradients = tf.reduce_mean(gradients, axis=0, keepdims=True)
    integrated_gradients = (image - baseline) * avg_gradients
    
    attribution = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)
    attribution = attribution[0]  
    
    return attribution.numpy()

def create_overlay(image, heatmap, alpha=0.4):
    """Create overlay of heatmap on image"""
    img = image[0].numpy()
    img = (img - img.min()) / (img.max() - img.min()) 
    
    heatmap_resized = tf.image.resize(
        tf.expand_dims(heatmap, -1), 
        (299, 299)
    ).numpy()[:, :, 0]
    
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    overlay = heatmap_colored * alpha + img * (1 - alpha)
    return overlay

def analyze_adversarial_example(image_path, epsilon=0.2, target_class=207):
    model = load_model()
    
    original_image = preprocess_image(image_path)
    
    orig_class, orig_conf, orig_idx = get_prediction(model, original_image)
    print(f"Original prediction: {orig_class} ({orig_conf:.2%} confidence)")
    
    adv_image = create_adversarial_example(model, original_image, target_class, epsilon)
    
    # Get adversarial predictions
    adv_class, adv_conf, adv_idx = get_prediction(model, adv_image)
    print(f"Adversarial prediction: {adv_class} ({adv_conf:.2%} confidence)")
    
    orig_gradcam = generate_gradcam(model, original_image, orig_idx)
    adv_gradcam = generate_gradcam(model, adv_image, adv_idx)
    
    orig_attribution = generate_integrated_gradients(model, original_image, orig_idx)
    adv_attribution = generate_integrated_gradients(model, adv_image, adv_idx)
    
    create_comparison_plot(
        original_image, adv_image, 
        orig_gradcam, adv_gradcam,
        orig_attribution, adv_attribution,
        orig_class, adv_class, orig_conf, adv_conf, epsilon
    )
    

def create_comparison_plot(orig_img, adv_img, orig_gradcam, adv_gradcam, 
                          orig_attr, adv_attr, orig_class, adv_class, 
                          orig_conf, adv_conf, epsilon):
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(orig_img[0] * 0.5 + 0.5)
    axes[0, 0].set_title(f'Original Image\n{orig_class}\n{orig_conf:.1%} Confidence', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    orig_overlay = create_overlay(orig_img, orig_gradcam)
    axes[0, 1].imshow(orig_overlay)
    axes[0, 1].set_title('Grad-CAM Heatmap\n(Important Areas)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(orig_attr, cmap='hot')
    axes[0, 2].set_title('Integrated Gradients\n(Attribution Mask)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(adv_img[0] * 0.5 + 0.5)
    axes[1, 0].set_title(f'Adversarial Image\n{adv_class}\n{adv_conf:.1%} Confidence\nε = {epsilon}', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    adv_overlay = create_overlay(adv_img, adv_gradcam)
    axes[1, 1].imshow(adv_overlay)
    axes[1, 1].set_title('Grad-CAM Heatmap\n(Confused Areas)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(adv_attr, cmap='hot')
    axes[1, 2].set_title('Integrated Gradients\n(Attribution Mask)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_dir = "adversarial_xai_results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"adversarial_comparison_epsilon_{epsilon}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()
    

if __name__ == "__main__":
    image_path = "./plane.jpeg" 
    
    analyze_adversarial_example(
        image_path=image_path,
        epsilon=0.2,
        target_class=207  
    )