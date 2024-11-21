import tensorflow as tf
import numpy as np
import json
import time
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Start the timer
start_time = time.time()

# Configurable variables
dataset_type = 'MNIST'  # Set this to either 'MNIST' or 'CIFAR10'

# Load the dataset based on the dataset_type variable
if dataset_type == 'MNIST':
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    input_shape = (28, 28, 1)
    x_train_full, x_test_full = x_train_full[..., np.newaxis] / \
        255.0, x_test_full[..., np.newaxis] / 255.0
elif dataset_type == 'CIFAR10':
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
    input_shape = (32, 32, 3)
    x_train_full, x_test_full = x_train_full / 255.0, x_test_full / 255.0

# Create data subsets
train_mask_8 = np.isin(y_train_full, np.arange(8)).flatten()
test_mask_8 = np.isin(y_test_full, np.arange(8)).flatten()
test_mask_9_10 = np.isin(y_test_full, [8, 9]).flatten()

x_train_8, y_train_8 = x_train_full[train_mask_8], y_train_full[train_mask_8]
x_test_8, y_test_8 = x_test_full[test_mask_8], y_test_full[test_mask_8]
x_test_9_10, y_test_9_10 = x_test_full[test_mask_9_10], y_test_full[test_mask_9_10]

# Training and validation splits for class 0-7 subsets
x_train, x_val = x_train_8[:int(
    0.8 * len(x_train_8))], x_train_8[int(0.8 * len(x_train_8)):]
y_train, y_val = y_train_8[:int(
    0.8 * len(y_train_8))], y_train_8[int(0.8 * len(y_train_8)):]

# Define the CNN model


def build_model(num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu',
               input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Initial training on classes 0-7
model_8_classes = build_model(8)
model_8_classes.compile(
    optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model_8_classes.summary())
history_8 = model_8_classes.fit(
    x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Evaluate the model trained on classes 0-7
evaluation_8_test_8 = model_8_classes.evaluate(x_test_8, y_test_8, verbose=0)
results_8_test_8 = {
    "loss": evaluation_8_test_8[0], "accuracy": evaluation_8_test_8[1]}

# Save results for initial training on classes 0-7
with open("results_8_classes.json", "w") as json_file:
    json.dump({"0-7 test set": results_8_test_8}, json_file)

# Modify the model for 10 classes, freeze initial layers, and initialize last two neurons
model_10_classes = build_model(10)
for layer, old_layer in zip(model_10_classes.layers[:-1], model_8_classes.layers[:-1]):
    layer.set_weights(old_layer.get_weights())
    layer.trainable = False

# Initialize last layer's weights for new neurons (classes 8 and 9) to zero
old_weights, old_biases = model_8_classes.layers[-1].get_weights()
new_weights = np.zeros((old_weights.shape[0], 10))
new_biases = np.zeros((10,))
new_weights[:, :8] = old_weights
new_biases[:8] = old_biases
model_10_classes.layers[-1].set_weights([new_weights, new_biases])

# Train the updated model on the full dataset with classes 0-9
model_10_classes.compile(
    optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model_10_classes.summary())
history_10 = model_10_classes.fit(
    x_train_full, y_train_full, epochs=10, validation_data=(x_val, y_val))

# Separate evaluations for different sets
evaluation_10_test_8 = model_10_classes.evaluate(x_test_8, y_test_8, verbose=0)
evaluation_10_test_9_10 = model_10_classes.evaluate(
    x_test_9_10, y_test_9_10, verbose=0)
evaluation_10_test_full = model_10_classes.evaluate(
    x_test_full, y_test_full, verbose=0)

# Save evaluations in JSON format
results_10 = {
    "0-7 test set": {"loss": evaluation_10_test_8[0], "accuracy": evaluation_10_test_8[1]},
    "8-9 test set": {"loss": evaluation_10_test_9_10[0], "accuracy": evaluation_10_test_9_10[1]},
    "full test set": {"loss": evaluation_10_test_full[0], "accuracy": evaluation_10_test_full[1]},
}

with open("results_10_classes.json", "w") as json_file:
    json.dump(results_10, json_file)

# End the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")

# Optionally, save the elapsed time to a JSON file
with open("time_report.json", "w") as json_file:
    json.dump({"total_time_seconds": elapsed_time}, json_file)

print("Training and evaluation complete. Results and time report saved to JSON files.")
