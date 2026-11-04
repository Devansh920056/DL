# 1. Import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

print("TensorFlow Version:", tf.__version__)

# 2. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Load pre-trained VGG16 model
base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(96, 96, 3)
)

# Freeze pre-trained layers
base_model.trainable = False

print("Pre-trained VGG16 model loaded.")
print("Base model layers are frozen.")

# 4. Build custom classification model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Resizing(96, 96),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train custom classifier
print("Training custom classifier...")

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 7. Fine-tune the pre-trained model
base_model.trainable = True

# Unfreeze only the last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning started...")

history_fine = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

# 8. Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# 9. Plot training and validation accuracy
train_acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

plt.figure(figsize=(8, 5))
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.axvline(x=4, linestyle="--", label="Fine-tuning starts")

plt.title("Transfer Learning using VGG16")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
