import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# Sample Corpus
sentences = [
    "the cat sits on the mat",
    "the dog sits on the rug",
    "the cat eats fish",
    "the dog eats meat",
    "a cat likes fish",
    "a dog likes meat",
    "the cat likes the mat",
    "the dog likes the rug"
]

# Tokenization and vocabulary creation
tokenized = [s.lower().split() for s in sentences]

vocabulary = sorted(set(w for s in tokenized for w in s))

word_to_id = {w: i for i, w in enumerate(vocabulary)}
id_to_word = {i: w for w, i in word_to_id.items()}

vocab_size = len(vocabulary)

print("Vocabulary:", vocabulary)
print("Vocabulary Size:", vocab_size)

# Context window size
window_size = 2
contexts, targets = [], []

for sentence in tokenized:
    ids = [word_to_id[w] for w in sentence]

    for i, target in enumerate(ids):
        context = ids[max(0, i-window_size):i] + \
                  ids[i+1:i+window_size+1]

        if context:
            contexts.append(context)
            targets.append(target)

# Padding
max_len = 2 * window_size

X = np.full((len(contexts), max_len), -1, dtype=np.int32)

for i, c in enumerate(contexts):
    X[i, :len(c)] = c

y = np.array(targets, dtype=np.int32)

print("Training Examples:", len(y))
print("Input Shape:", X.shape)
print("Target Shape:", y.shape)

# Define model
embedding_dim = 16

inputs = layers.Input(shape=(max_len,))

safe_ids = layers.Lambda(
    lambda x: tf.maximum(x, 0)
)(inputs)

embedding = layers.Embedding(vocab_size, embedding_dim)(safe_ids)

# Mask padding values
mask = layers.Lambda(
    lambda x: tf.cast(x != -1, tf.float32)[..., None]
)(inputs)

masked = layers.Multiply()([embedding, mask])

# Average context embeddings
average = layers.Lambda(
    lambda x: tf.reduce_sum(x[0], axis=1) /
    tf.maximum(tf.reduce_sum(x[1], axis=1), 1.0)
)([masked, mask])

outputs = layers.Dense(vocab_size, activation="softmax")(average)

model = Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    X, y,
    epochs=150,
    batch_size=16,
    verbose=1
)

print("Training Completed!")
print("Final Accuracy:",
      round(history.history["accuracy"][-1] * 100, 2), "%")

# Predict target word
def predict_target(words):

    ids = [word_to_id[w] for w in words if w in word_to_id]

    padded = np.full((1, max_len), -1, dtype=np.int32)
    padded[0, :min(len(ids), max_len)] = ids[:max_len]

    prediction = model.predict(padded, verbose=0)[0]

    top_words = np.argsort(prediction)[::-1][:5]

    print("Context Words:", words)
    print("Predicted Words:")

    for i in top_words:
        print(id_to_word[i], ":", round(float(prediction[i]), 4))


predict_target(["cat", "on", "mat"])

# Display training loss
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.title("CBOW Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
