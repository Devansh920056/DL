import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# -----------------------------------------
# 1. DATA PREPARATION
# -----------------------------------------

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

# Tokenize sentences
tokenized_sentences = [s.lower().split() for s in sentences]

# Create vocabulary
vocabulary = sorted(set(
    word for sentence in tokenized_sentences
    for word in sentence
))

word_to_id = {word: i for i, word in enumerate(vocabulary)}
id_to_word = {i: word for word, i in word_to_id.items()}

vocab_size = len(vocabulary)

print("Vocabulary:", vocabulary)
print("Vocabulary Size:", vocab_size)


# -----------------------------------------
# 2. GENERATE TRAINING DATA
# -----------------------------------------

window_size = 2
contexts = []
targets = []

for sentence in tokenized_sentences:

    ids = [word_to_id[word] for word in sentence]

    for i, target in enumerate(ids):

        left = ids[max(0, i-window_size):i]
        right = ids[i+1:i+window_size+1]

        context = left + right

        if context:
            contexts.append(context)
            targets.append(target)

# Padding contexts to fixed length
max_len = 2 * window_size

X = np.full((len(contexts), max_len), -1, dtype=np.int32)

for i, context in enumerate(contexts):
    X[i, :len(context)] = context

y = np.array(targets, dtype=np.int32)

print("\nTraining Examples:", len(y))
print("Input Shape:", X.shape)
print("Target Shape:", y.shape)

# Display sample training pairs
print("\nSample Context-Target Pairs:")

for i in range(5):
    words = [id_to_word[j] for j in contexts[i]]
    print(words, "->", id_to_word[y[i]])


# -----------------------------------------
# 3. BUILD AND TRAIN CBOW MODEL
# -----------------------------------------

embedding_dim = 16

input_layer = layers.Input(shape=(max_len,))

# Replace padding values for embedding lookup
safe_ids = layers.Lambda(
    lambda x: tf.maximum(x, 0)
)(input_layer)

# Embedding layer
embedding = layers.Embedding(
    vocab_size, embedding_dim
)(safe_ids)

# Create mask for padding values
mask = layers.Lambda(
    lambda x: tf.cast(x != -1, tf.float32)[..., None]
)(input_layer)

# Remove padded embeddings
masked = layers.Multiply()([embedding, mask])

# Calculate average context embedding
average = layers.Lambda(
    lambda x: tf.reduce_sum(x[0], axis=1) /
    tf.maximum(tf.reduce_sum(x[1], axis=1), 1.0)
)([masked, mask])

# Output layer
output_layer = layers.Dense(
    vocab_size, activation="softmax"
)(average)

# Create model
model = Model(input_layer, output_layer)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# Train model
history = model.fit(
    X, y,
    epochs=150,
    batch_size=16,
    verbose=0
)

print("\nTraining Completed Successfully!")
print("Final Accuracy:",
      round(history.history["accuracy"][-1] * 100, 2), "%")


# -----------------------------------------
# 4. DISPLAY OUTPUT AND PREDICTION
# -----------------------------------------

def predict_target(context_words):

    ids = [
        word_to_id[w.lower()]
        for w in context_words
        if w.lower() in word_to_id
    ]

    if not ids:
        print("No known context words!")
        return

    ids = ids[:max_len]

    padded = np.full((1, max_len), -1, dtype=np.int32)
    padded[0, :len(ids)] = ids

    probabilities = model.predict(padded, verbose=0)[0]

    top_words = np.argsort(probabilities)[::-1][:5]

    print("\nContext Words:", context_words)
    print("Predicted Target Words:")

    for word_id in top_words:
        print(
            id_to_word[int(word_id)],
            "Probability:",
            round(float(probabilities[word_id]), 4)
        )


# Test prediction
predict_target(["cat", "on", "mat"])

# Display sample training predictions
print("\nSample Training Predictions:")

for i in range(5):

    prediction = model.predict(X[i:i+1], verbose=0)[0]
    predicted_id = np.argmax(prediction)

    print(
        "Actual:", id_to_word[y[i]],
        "| Predicted:", id_to_word[predicted_id]
    )

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss")

plt.title("CBOW Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
