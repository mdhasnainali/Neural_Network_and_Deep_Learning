import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets, applications
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Load Data ---
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values for the small CNN
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode labels for hard target loss
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

print(f"Dataset loaded. Training images shape: {train_images.shape}")

# --- 2. Load Models ---
# Load the small CNN (student) model
small_cnn_model_path = 'small_cnn_model.keras'
print(f"Loading Student Model (Small CNN) from: {small_cnn_model_path}")
student_model = models.load_model(small_cnn_model_path)
print("Student Model loaded successfully.")
student_model.summary()

# Load the VGG16 fine-tuned model (teacher)
vgg16_finetuned_model_path = 'vgg16_finetuned.keras'
print(f"Loading Teacher Model (VGG16 Fine-tuned) from: {vgg16_finetuned_model_path}")
teacher_model = models.load_model(vgg16_finetuned_model_path)
# Ensure the teacher model is not trainable during distillation
teacher_model.trainable = False
print("Teacher Model loaded successfully.")
teacher_model.summary()

# --- 3. Implement Knowledge Distillation ---

# Define the Temperature parameter
temperature = 3.0 # A hyperparameter to control the softness of teacher probabilities

class Distiller(models.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha # Weight for student_loss_fn (1-alpha for distillation_loss_fn)

    def train_step(self, data):
        x, y = data # x is images, y is hard labels

        # Convert CIFAR-10 images to 32x32 for the teacher model
        # and preprocess them as the teacher expects
        x_teacher = tf.image.resize(x, (32, 32))
        x_teacher = applications.vgg16.preprocess_input(x_teacher * 255.0) # Apply preprocessing for teacher (undo 0-1 norm)

        with tf.GradientTape() as tape:
            # Forward pass of teacher
            teacher_predictions = self.teacher(x_teacher, training=False)

            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Calculate hard targets loss (student vs. true labels)
            student_loss = self.student_loss_fn(y, student_predictions)

            # Calculate distillation loss
            # Soften teacher and student predictions with temperature
            teacher_soft_targets = tf.nn.softmax(teacher_predictions / temperature, axis=1)
            student_soft_targets = tf.nn.softmax(student_predictions / temperature, axis=1)

            # Compute distillation loss (KL Divergence is often used, but CC with soft targets works)
            distillation_loss = self.distillation_loss_fn(teacher_soft_targets, student_soft_targets)
            distillation_loss *= (temperature**2) # Scale loss by T^2 as per original paper

            # Combine losses
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients and apply
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (e.g., accuracy for student)
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results["student_loss"] = student_loss
        results["distillation_loss"] = distillation_loss
        return results

    def test_step(self, data):
        # Evaluation is standard: just evaluate the student against hard labels
        x, y = data
        student_predictions = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_predictions)
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results["student_loss"] = student_loss
        return results


# Initialize and compile the distiller
distiller = Distiller(student=student_model, teacher=teacher_model)

distiller.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001), # Use a low learning rate
    metrics=['accuracy'],
    student_loss_fn=losses.CategoricalCrossentropy(from_logits=False),
    distillation_loss_fn=losses.CategoricalCrossentropy(from_logits=False), # Note: from_logits=False because we softmaxed
    alpha=0.5 # Balance between hard and soft targets (0.5 means equal weight)
)

# --- 4. Train the Distilled Model ---
print("\n--- Training the Student Model with Knowledge Distillation from VGG16 ---")
distiller_history = distiller.fit(train_images, train_labels_one_hot,
                                  epochs=20, # More epochs might be needed for distillation
                                  validation_data=(test_images, test_labels_one_hot),
                                  batch_size=64)

# --- 5. Evaluate the Distilled Model ---
print("\nEvaluating the distilled student model...")
test_loss_distilled, test_acc_distilled = distiller.evaluate(test_images, test_labels_one_hot, verbose=2)
print(f"Test accuracy of Small CNN (distilled from VGG16): {test_acc_distilled:.4f}")

# Optionally save the distilled student model (it's the student_model object within distiller)
distilled_model_save_path = 'small_cnn_distilled_vgg16.keras'
print(f"Saving the distilled Small CNN model to: {distilled_model_save_path}")
student_model.save(distilled_model_save_path)
print("Distilled Small CNN model saved successfully.")




