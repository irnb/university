import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sn

# Set random seed for reproducibility
SEED_VALUE = 72
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Load MNIST dataset
(X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()
X_valid = X_train_all[:10000]
X_train = X_train_all[10000:]
y_valid = y_train_all[:10000]
y_train = y_train_all[10000:]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

# Plot example images
plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.axis("off")
    plt.imshow(X_train[i], cmap='gray')
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

# Normalize and reshape data
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype("float32") / 255
X_valid = X_valid.reshape((X_valid.shape[0], 28 * 28)).astype("float32") / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype("float32") / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
training_results = model.fit(X_train, y_train, epochs=21, batch_size=64, validation_data=(X_valid, y_valid))

# Plot results
def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))
    if not isinstance(metric_name, (list, tuple)):
        metrics = [metrics]
        metric_name = [metric_name]
    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, len(metrics[0]) - 1])
    if ylim:
        plt.ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()

train_loss = training_results.history["loss"]
train_acc = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc = training_results.history["val_accuracy"]

plot_results([train_loss, valid_loss], ylabel="Loss", ylim=[0.0, 0.5],
             metric_name=["Training Loss", "Validation Loss"], color=["g", "b"])
plot_results([train_acc, valid_acc], ylabel="Accuracy", ylim=[0.9, 1.0],
             metric_name=["Training Accuracy", "Validation Accuracy"], color=["g", "b"])

# Predictions
predictions = model.predict(X_test)

index = 0
print('Ground truth for test digit:', np.argmax(y_test[index]))
print('\nPredictions for each class:\n')
for i in range(10):
    print(f'digit: {i}, probability: {predictions[index][i]}')

# Confusion matrix
predicted_labels = np.argmax(predictions, axis=1)
y_test_integer_labels = np.argmax(y_test, axis=1)
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

plt.figure(figsize=[15, 8])
sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 14})
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Display misclassified examples
CLASSES = np.arange(0, 10)
actual_single = CLASSES[y_test_integer_labels]
preds_single = CLASSES[predicted_labels]
wrong_indices = np.where(preds_single != actual_single)[0]

n_to_show = min(10, len(wrong_indices))
indices = np.random.choice(wrong_indices, n_to_show, replace=False)

fig = plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f'P: {preds_single[idx]} | A: {actual_single[idx]}')
plt.show()