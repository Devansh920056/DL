# 1. Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist


# 2. Load and Preprocess Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Use digit 0 as normal data
x_train_normal = x_train[y_train == 0]

# Identify anomalous test images
y_test_anomaly = (y_test != 0).astype("int32")


# 3. Build Encoder
input_layer = layers.Input(shape=(784,))

x = layers.Dense(128, activation="relu")(input_layer)
x = layers.Dense(64, activation="relu")(x)

encoded = layers.Dense(32, activation="relu")(x)

encoder = Model(input_layer, encoded)


# 4. Build Decoder
decoder_input = layers.Input(shape=(32,))

x = layers.Dense(64, activation="relu")(decoder_input)
x = layers.Dense(128, activation="relu")(x)

decoded = layers.Dense(784, activation="sigmoid")(x)

decoder = Model(decoder_input, decoded)


# Combine Encoder and Decoder
autoencoder_input = layers.Input(shape=(784,))
output = decoder(encoder(autoencoder_input))

autoencoder = Model(autoencoder_input, output)


# 5. Compile the Model
autoencoder.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# 6. Train the Autoencoder
autoencoder.fit(
    x_train_normal,
    x_train_normal,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)


# 7. Calculate Reconstruction Error
reconstructed = autoencoder.predict(x_test)

error = np.mean(
    np.square(x_test - reconstructed), axis=1
)

# Calculate threshold using normal training images
normal_reconstructed = autoencoder.predict(x_train_normal)

normal_error = np.mean(
    np.square(x_train_normal - normal_reconstructed), axis=1
)

threshold = np.percentile(normal_error, 95)

print("Anomaly Threshold:", threshold)


# 8. Detect Anomalies
predictions = (error > threshold).astype("int32")

accuracy = np.mean(predictions == y_test_anomaly)

print("Detection Accuracy:", accuracy * 100, "%")
print("Detected Anomalies:", np.sum(predictions))


# 9. Display Original and Reconstructed Images
plt.figure(figsize=(10, 4))

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
