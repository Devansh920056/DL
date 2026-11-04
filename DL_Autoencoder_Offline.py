# 1. Import Required Libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Model, layers

print("TensorFlow Version:", tf.__version__)


# 2. Load and Preprocess MNIST Dataset (Offline)

data = np.load("mnist.npz")

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784 features
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Select digit 0 as normal training data
x_train_normal = x_train[y_train == 0]

# Label normal and anomalous test samples
y_test_anomaly = (y_test != 0).astype("int32")

print("Normal Training Samples:", x_train_normal.shape)
print("Test Samples:", x_test.shape)
print("Anomalous Samples:", np.sum(y_test_anomaly))


# 3. Build Encoder Network

input_dim = 784
latent_dim = 32

encoder_input = layers.Input(shape=(input_dim,))

x = layers.Dense(128, activation="relu")(encoder_input)
x = layers.Dense(64, activation="relu")(x)

latent_vector = layers.Dense(
    latent_dim, activation="relu"
)(x)

encoder = Model(encoder_input, latent_vector, name="Encoder")

encoder.summary()


# 4. Build Decoder Network

decoder_input = layers.Input(shape=(latent_dim,))

x = layers.Dense(64, activation="relu")(decoder_input)
x = layers.Dense(128, activation="relu")(x)

decoder_output = layers.Dense(
    input_dim, activation="sigmoid"
)(x)

decoder = Model(decoder_input, decoder_output, name="Decoder")

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


# 7. Evaluate the Model

test_loss, test_mae = autoencoder.evaluate(
    x_test, x_test, verbose=0
)

print("\nTest Reconstruction Loss:", test_loss)
print("Test Reconstruction MAE:", test_mae)


# 8. Calculate Reconstruction Error and Threshold

x_test_reconstructed = autoencoder.predict(
    x_test, verbose=0
)

reconstruction_error = np.mean(
    np.square(x_test - x_test_reconstructed), axis=1
)

# Calculate normal training reconstruction error
normal_reconstructed = autoencoder.predict(
    x_train_normal, verbose=0
)

normal_error = np.mean(
    np.square(x_train_normal - normal_reconstructed), axis=1
)

# Set threshold using 95th percentile
threshold = np.percentile(normal_error, 95)

print("Anomaly Threshold:", threshold)


# 9. Detect Anomalies

y_pred_anomaly = (
    reconstruction_error > threshold
).astype("int32")

print("Detected Anomalies:", np.sum(y_pred_anomaly == 1))


# 10. Calculate Evaluation Metrics

TP = np.sum((y_pred_anomaly == 1) & (y_test_anomaly == 1))
TN = np.sum((y_pred_anomaly == 0) & (y_test_anomaly == 0))
FP = np.sum((y_pred_anomaly == 1) & (y_test_anomaly == 0))
FN = np.sum((y_pred_anomaly == 0) & (y_test_anomaly == 1))

accuracy = (TP + TN) / len(y_test_anomaly)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)

print("\nDetection Accuracy:", accuracy * 100, "%")
print("Precision:", precision)
print("Recall:", recall)


# 11. Plot Training and Validation Loss

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 12. Display Original and Reconstructed Images

sample_indices = []

for label in [0, 1]:
    candidates = np.where(y_test_anomaly == label)[0]
    sample_indices.extend(candidates[:5].tolist())

plt.figure(figsize=(12, 5))

for i, idx in enumerate(sample_indices):

    # Original Image
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed Image
    plt.subplot(2, 10, i + 11)
    plt.imshow(
        x_test_reconstructed[idx].reshape(28, 28),
        cmap="gray"
    )

    result = "Anomaly" if y_pred_anomaly[idx] else "Normal"
    plt.title(result)
    plt.axis("off")

plt.tight_layout()
plt.show()
