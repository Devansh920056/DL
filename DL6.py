
# 1. Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

print("TensorFlow Version:", tf.__version__)

# 2. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]

# Use subset for faster training
x_train, y_train = x_train[:10000], y_train[:10000].flatten()
x_test, y_test = x_test[:2000], y_test[:2000].flatten()

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)

# 3. Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pre-trained convolutional layers
base_model.trainable = False

print("MobileNetV2 loaded successfully.")
print("Pre-trained layers are frozen.")

# 4. Build custom classification model
inputs = layers.Input(shape=(32, 32, 3))

# Data augmentation and resizing
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.Resizing(96, 96)(x)

# Preprocess input for MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Extract features using pre-trained CNN
x = base_model(x, training=False)

# Add custom trainable classifier
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = models.Model(inputs, outputs)

# 5. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train custom classifier layers
print("Training custom classifier...")

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32,
    verbose=1
)

# 7. Fine-tune the pre-trained CNN
base_model.trainable = True

# Unfreeze only the top 20% layers
fine_tune_at = int(len(base_model.layers) * 0.8)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Recompile using a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning started...")

history_fine = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=32,
    verbose=1
)

# 8. Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# 9. Predict test images
predictions = model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

# Display actual and predicted classes
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(
        f"Pred: {class_names[predicted_labels[i]]}\n"
        f"Actual: {class_names[y_test[i]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# 10. Plot training and validation accuracy
train_acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

plt.figure(figsize=(8, 5))
plt.plot(train_acc, label="Training Accuracy", marker="o")
plt.plot(val_acc, label="Validation Accuracy", marker="o")
plt.axvline(5, linestyle="--", label="Fine-tuning starts")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Transfer Learning using MobileNetV2")
plt.legend()
plt.grid(True)
plt.show()
