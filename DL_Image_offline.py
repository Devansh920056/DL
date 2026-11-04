import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

print("TensorFlow Version:", tf.__version__)

# Load CIFAR-10 dataset from local files

data_path = "cifar-10-batches-py"

def load_batch(file):
    with open(os.path.join(data_path, file), "rb") as f:
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
    images, labels = load_batch("data_batch_" + str(i))
    x_train_list.append(images)
    y_train_list.append(labels)

x_train = np.concatenate(x_train_list)
y_train = np.concatenate(y_train_list)

# Load testing batch
x_test, y_test = load_batch("test_batch")

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# Normalize pixel values between 0 and 1

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels into categorical format

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Display sample images

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(
    x_test, y_test_cat
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

predictions = model.predict(x_test[:10])

predicted_labels = np.argmax(predictions, axis=1)
actual_labels = y_test[:10]

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i])

    predicted = class_names[predicted_labels[i]]
    actual = class_names[actual_labels[i]]

    color = "green" if predicted_labels[i] == actual else "red"

    plt.title("Pred: " + predicted + "\nActual: " + actual,
              color=color, fontsize=9)

    plt.axis("off")

plt.tight_layout()
plt.show()

# Training and validation accuracy

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Training and validation loss

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
