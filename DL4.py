# 1. Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist

print("TensorFlow Version:", tf.__version__)


# 2. Load and Preprocess Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images into 784 features
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Select only digit 0 as normal training data
x_train_normal = x_train[y_train == 0]

# Identify normal and anomalous test samples
y_test_anomaly = (y_test != 0).astype("int32")

print("Normal Training Samples:", x_train_normal.shape)
print("Testing Samples:", x_test.shape)


# 3. Build Encoder Network
input_dim = 784
latent_dim = 32

encoder_input = layers.Input(shape=(input_dim,))

x = layers.Dense(128, activation="relu")(encoder_input)
x = layers.Dense(64, activation="relu")(x)

latent_vector = layers.Dense(
    latent_dim, activation="relu"
)(x)

encoder = Model(
    encoder_input, latent_vector, name="Encoder"
)

encoder.summary()


# 4. Build Decoder Network
decoder_input = layers.Input(shape=(latent_dim,))

x = layers.Dense(64, activation="relu")(decoder_input)
x = layers.Dense(128, activation="relu")(x)

decoder_output = layers.Dense(
    input_dim, activation="sigmoid"
)(x)

decoder = Model(
    decoder_input, decoder_output, name="Decoder"
)

decoder.summary()


# Combine Encoder and Decoder
autoencoder_input = layers.Input(shape=(input_dim,))

encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

autoencoder = Model(
    autoencoder_input, decoded, name="Autoencoder"
)

autoencoder.summary()


# 5. Compile the Autoencoder
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)


# 6. Train the Autoencoder
history = autoencoder.fit(
    x_train_normal,
    x_train_normal,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)


# 7. Evaluate the Autoencoder
test_loss, test_mae = autoencoder.evaluate(
    x_test, x_test, verbose=0
)

print("\nTest Reconstruction Loss:", test_loss)
print("Test Reconstruction MAE:", test_mae)


# 8. Calculate Reconstruction Error
x_test_reconstructed = autoencoder.predict(
    x_test, verbose=0
)

reconstruction_error = np.mean(
    np.square(x_test - x_test_reconstructed),
    axis=1
)


# Calculate Threshold Using Normal Training Data
normal_reconstructed = autoencoder.predict(
    x_train_normal, verbose=0
)

normal_error = np.mean(
    np.square(x_train_normal - normal_reconstructed),
    axis=1
)

threshold = np.percentile(normal_error, 95)

print("Anomaly Threshold:", threshold)


# 9. Detect Anomalies
y_pred_anomaly = (
    reconstruction_error > threshold
).astype("int32")

print("Detected Anomalies:",
      np.sum(y_pred_anomaly == 1))


# Calculate Evaluation Metrics
accuracy = np.mean(
    y_pred_anomaly == y_test_anomaly
)

true_positive = np.sum(
    (y_pred_anomaly == 1) & (y_test_anomaly == 1)
)

false_positive = np.sum(
    (y_pred_anomaly == 1) & (y_test_anomaly == 0)
)

false_negative = np.sum(
    (y_pred_anomaly == 0) & (y_test_anomaly == 1)
)

precision = true_positive / (true_positive + false_positive + 1e-8)

recall = true_positive / (true_positive + false_negative + 1e-8)

print("\nDetection Accuracy:", accuracy * 100, "%")
print("Precision:", precision)
print("Recall:", recall)


# 10. Plot Training and Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")

plt.legend()
plt.grid(True)
plt.show()


# 11. Display Original and Reconstructed Images
plt.figure(figsize=(12, 5))

for i in range(5):

    # Original Image
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed Image
    plt.subplot(2, 5, i + 6)
    plt.imshow(
        x_test_reconstructed[i].reshape(28, 28),
        cmap="gray"
    )
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
