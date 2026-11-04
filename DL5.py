import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. DATA PREPARATION

text = "the cat sits on the mat the dog plays in the park"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

words = text.split()

word_to_index = tokenizer.word_index
index_to_word = {v: k for k, v in word_to_index.items()}
vocab_size = len(word_to_index) + 1

print("Vocabulary:", word_to_index)
print("Vocabulary Size:", vocab_size)


# 2. GENERATE TRAINING DATA

window = 2
X, y = [], []

for i in range(window, len(words) - window):

    context = words[i-window:i] + words[i+1:i+window+1]
    target = words[i]

    X.append([word_to_index[w] for w in context])
    y.append(word_to_index[target])

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

print("\nInput Shape:", X.shape)
print("Target Shape:", y.shape)


# 3. BUILD AND TRAIN CBOW MODEL

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size, 50, input_length=window*2
    ),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X, y,
    epochs=200,
    verbose=0
)

print("\nTraining Completed Successfully!")
print("Final Accuracy:",
      round(history.history["accuracy"][-1] * 100, 2), "%")


# 4. DISPLAY OUTPUT AND PREDICTION

context_words = ["the", "on", "the", "mat"]

context = np.array([[
    word_to_index[w] for w in context_words
]])

prediction = model.predict(context, verbose=0)

predicted_index = np.argmax(prediction)
predicted_word = index_to_word[predicted_index]

print("\nContext Words:", context_words)
print("Predicted Target Word:", predicted_word)


# Display sample predictions

print("\nSample Training Predictions:")

for i in range(min(5, len(X))):
    prediction = model.predict(X[i:i+1], verbose=0)
    predicted_index = np.argmax(prediction)

    print("Actual:", index_to_word[
        int(np.argmax(y[i]))
    ], "| Predicted:", index_to_word[predicted_index])


# Plot training loss

plt.plot(history.history["loss"], label="Training Loss")
plt.title("CBOW Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
