import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset folder path
data_dir = Path("cifar-10-batches-py")

# Function to load a CIFAR-10 batch
def load_batch(file_path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, np.array(labels)

# Load training batches
x_train_list = []
y_train_list = []

for i in range(1, 6):
    images, labels = load_batch(data_dir / f"data_batch_{i}")
    x_train_list.append(images)
    y_train_list.append(labels)

# Combine training data
x_train = np.concatenate(x_train_list)
y_train = np.concatenate(y_train_list)

# Load testing data
x_test, y_test = load_batch(data_dir / "test_batch")

# Use a subset for faster training
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Local path of pretrained weights
weights_path = Path("models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5")

# Check whether weights exist
if not weights_path.exists():
    raise FileNotFoundError("MobileNetV2 weights not found!")

# Load MobileNetV2 architecture without classification head
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights=None
)

# Load pretrained ImageNet weights locally
base_model.load_weights(str(weights_path))

# Freeze pretrained layers
base_model.trainable = False

print("MobileNetV2 loaded successfully!")
print("Pretrained layers are frozen.")

# Define input layer
inputs = layers.Input(shape=(32, 32, 3))

# Data augmentation
x = layers.RandomFlip("horizontal")(inputs)
x = layers.Resizing(96, 96)(x)

# Preprocess images for MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Extract features using pretrained CNN
x = base_model(x, training=False)

# Custom classification layers
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Output layer
outputs = layers.Dense(10, activation="softmax")(x)

# Create model
model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32,
    verbose=1
)

print("Initial training completed!")

# Unfreeze pretrained model
base_model.trainable = True

# Unfreeze only the top 20% layers
fine_tune_at = int(len(base_model.layers) * 0.8)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Compile with smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Fine-tune model
history_fine = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32,
    verbose=1
)

print("Fine-tuning completed!")

test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

predictions = model.predict(x_test[:10], verbose=0)

predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i])

    color = "green" if predicted_labels[i] == y_test[i] else "red"

    plt.title(
        f"Pred: {class_names[predicted_labels[i]]}\n"
        f"Actual: {class_names[y_test[i]]}",
        color=color,
        fontsize=8
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

train_acc = (
    history.history["accuracy"] +
    history_fine.history["accuracy"]
)

val_acc = (
    history.history["val_accuracy"] +
    history_fine.history["val_accuracy"]
)

plt.figure(figsize=(8, 5))

plt.plot(train_acc, marker="o", label="Training Accuracy")
plt.plot(val_acc, marker="o", label="Validation Accuracy")

plt.axvline(5, linestyle="--", label="Fine-tuning Starts")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Transfer Learning using MobileNetV2")

plt.legend()
plt.grid(True)
plt.show()
