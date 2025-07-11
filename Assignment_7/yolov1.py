import tensorflow as tf
import numpy as np
import os
import cv2
from keras import layers, models

# ========== CONFIGURATION ==========
IMG_SIZE = 448
GRID_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 1
BATCH_SIZE = 16
EPOCHS = 50

# ========== HELPER FUNCTIONS ==========

def parse_annotation_line(line):
    parts = list(map(float, line.strip().split()))
    class_id = int(parts[0])
    x_center, y_center, width, height = parts[1:]
    return [x_center, y_center, width, height, 1.0] + [0.0] * NUM_CLASSES if class_id == 0 else [0.0] * (5 + NUM_CLASSES)

def load_data(image_dir, label_dir):
    images, labels = [], []
    for img_name in os.listdir(image_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        
        label_grid = np.zeros((GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    col = int(x * GRID_SIZE)
                    row = int(y * GRID_SIZE)
                    x_cell = x * GRID_SIZE - col
                    y_cell = y * GRID_SIZE - row
                    w_cell = w * GRID_SIZE
                    h_cell = h * GRID_SIZE
                    label_grid[row, col] = [x_cell, y_cell, w_cell, h_cell, 1.0] + [1.0] * NUM_CLASSES
        
        images.append(img)
        labels.append(label_grid)
    
    return np.array(images), np.array(labels)

# ========== MODEL DEFINITION ==========

def build_yolo_v1_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(192, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (1, 1), activation="relu")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (1, 1), activation="relu")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(GRID_SIZE * GRID_SIZE * (5 + NUM_CLASSES))(x)
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES))(x)

    return tf.keras.Model(inputs, outputs)

# ========== CUSTOM YOLOv1 LOSS FUNCTION ==========

def custom_yolo_loss(y_true, y_pred):
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    obj_mask = y_true[..., 4:5]
    noobj_mask = 1.0 - obj_mask

    loc_loss = lambda_coord * tf.reduce_sum(obj_mask * tf.square(y_true[..., 0:4] - y_pred[..., 0:4]))

    obj_conf_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))
    noobj_conf_loss = lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))

    class_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., 5:] - y_pred[..., 5:]))

    total_loss = loc_loss + obj_conf_loss + noobj_conf_loss + class_loss
    return total_loss

# ========== TRAINING PIPELINE ==========

def train():
    image_dir = "./yolo_format_data/images/val"   
    label_dir = "./yolo_format_data/labels/val"   

    x_train, y_train = load_data(image_dir, label_dir)
    print(f"Loaded {len(x_train)} training images.")

    model = build_yolo_v1_model()
    model.compile(optimizer='adam', loss=custom_yolo_loss)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save("yolov1_tf_face_model.h5")

# ========== EVALUATION ==========

def predict_single_image(model, image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    pred = model.predict(img_input)[0]  # (7, 7, 6)
    boxes = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell = pred[i, j]
            confidence = cell[4]
            if confidence > 0.5:
                x, y, w, h = cell[0], cell[1], cell[2], cell[3]
                x_center = (j + x) * IMG_SIZE / GRID_SIZE
                y_center = (i + y) * IMG_SIZE / GRID_SIZE
                box_width = w * IMG_SIZE / GRID_SIZE
                box_height = h * IMG_SIZE / GRID_SIZE

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                boxes.append((x1, y1, x2, y2, confidence))

    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Conf: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("YOLOv1 Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== RUN ==========

if __name__ == "__main__":
    train()
    # model = tf.keras.models.load_model("yolov1_tf_face_model.h5", custom_objects={'custom_yolo_loss': custom_yolo_loss})
    # predict_single_image(model, "images/example.jpg")
