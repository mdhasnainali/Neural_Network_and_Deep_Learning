import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for binary segmentation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union) metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Load model with custom objects
model = load_model(
    'best_unet_flood_model.keras',
    custom_objects={
        'dice_loss': dice_loss,
        'dice_coefficient': dice_coefficient,
        'iou_metric': iou_metric
    }
)

def predict_and_save(image_path, save_dir, model, img_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    original_size = (image.shape[1], image.shape[0])  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, img_size)
    input_image = image_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_image, axis=0)

    pred_mask = model.predict(input_tensor)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST) * 255

    mask_path = os.path.join(save_dir, "predicted_mask.png")
    cv2.imwrite(mask_path, pred_mask_resized)

    overlay = cv2.addWeighted(image, 0.7, cv2.applyColorMap(pred_mask_resized, cv2.COLORMAP_JET), 0.3, 0)
    overlay_path = os.path.join(save_dir, "overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"Saved predicted mask to {mask_path}")
    print(f"Saved overlay to {overlay_path}")

predict_and_save(
    image_path="./archive/Image/1.jpg",
    save_dir="./results/",
    model=model
)

