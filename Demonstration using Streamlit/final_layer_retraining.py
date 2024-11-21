# final_layer_retraining.py

import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.callbacks import Callback


class EpochLogger(Callback):
    """Custom callback to log epoch details for each epoch."""

    def __init__(self):
        self.epoch_logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_logs.append(
            {"epoch": epoch + 1, "loss": logs["loss"], "accuracy": logs["accuracy"]})


def prepare_tf_dataset(data, dataset_name):
    images = np.array([np.array(img[0].numpy().transpose(1, 2, 0))
                      for img in data])
    labels = np.array([img[1] for img in data])
    if dataset_name == "MNIST":
        images = images.reshape(-1, 28, 28, 1)
    images = images / 255.0  # Normalize
    return images, labels


def build_model(num_classes, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_and_evaluate_final_layer_retraining(train_data, test_data, dataset_name):
    # Prepare data
    x_train_full, y_train_full = prepare_tf_dataset(train_data, dataset_name)
    x_test_full, y_test_full = prepare_tf_dataset(test_data, dataset_name)

    # Select initial classes (0-7) for training
    mask_8 = np.isin(y_train_full, np.arange(8))
    x_train_8, y_train_8 = x_train_full[mask_8], y_train_full[mask_8]

    # Determine input shape based on dataset
    input_shape = (28, 28, 1) if dataset_name == "MNIST" else (32, 32, 3)

    # Train initial model on classes 0-7
    model_8_classes = build_model(8, input_shape)
    model_8_classes.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Capture model summary of the initial model
    initial_model_summary = []
    model_8_classes.summary(print_fn=lambda x: initial_model_summary.append(x))

    # Train the model with epoch logging
    epoch_logger_8 = EpochLogger()
    start_time = time.time()
    history_8 = model_8_classes.fit(x_train_8, y_train_8, epochs=5, callbacks=[
                                    epoch_logger_8], verbose=1)
    initial_train_time = time.time() - start_time

    # Evaluate initial model on 0-7 test data
    evaluation_8_test_8 = model_8_classes.evaluate(
        x_test_full[y_test_full < 8], y_test_full[y_test_full < 8], verbose=0)

    # Modify model for 10 classes and freeze initial layers
    model_10_classes = build_model(10, input_shape)
    for layer, old_layer in zip(model_10_classes.layers[:-1], model_8_classes.layers[:-1]):
        layer.set_weights(old_layer.get_weights())
        layer.trainable = False

    # Initialize last layer with weights for classes 0-7 and zeros for new classes 8 and 9
    old_weights, old_biases = model_8_classes.layers[-1].get_weights()
    new_weights = np.zeros((old_weights.shape[0], 10))
    new_biases = np.zeros(10)
    new_weights[:, :8] = old_weights
    new_biases[:8] = old_biases
    model_10_classes.layers[-1].set_weights([new_weights, new_biases])

    # Capture model summary of the final model
    final_model_summary = []
    model_10_classes.summary(print_fn=lambda x: final_model_summary.append(x))

    # Compile and train final layer on full dataset
    model_10_classes.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    epoch_logger_10 = EpochLogger()
    start_time = time.time()
    history_10 = model_10_classes.fit(
        x_train_full, y_train_full, epochs=5, callbacks=[epoch_logger_10], verbose=1)
    final_train_time = time.time() - start_time

    # Evaluate final model on different subsets
    evaluation_10_test_8 = model_10_classes.evaluate(
        x_test_full[y_test_full < 8], y_test_full[y_test_full < 8], verbose=0)
    evaluation_10_test_9_10 = model_10_classes.evaluate(x_test_full[(y_test_full == 8) | (
        y_test_full == 9)], y_test_full[(y_test_full == 8) | (y_test_full == 9)], verbose=0)
    evaluation_10_test_full = model_10_classes.evaluate(
        x_test_full, y_test_full, verbose=0)

    # Return all results and summaries for display
    results = {
        "initial_model_summary": "\n".join(initial_model_summary),
        "initial_training_epochs": epoch_logger_8.epoch_logs,
        "initial_train_time": initial_train_time,
        "initial_evaluation_0_7": {"loss": evaluation_8_test_8[0], "accuracy": evaluation_8_test_8[1]},

        "final_model_summary": "\n".join(final_model_summary),
        "final_training_epochs": epoch_logger_10.epoch_logs,
        "final_train_time": final_train_time,
        "final_evaluation_0_7": {"loss": evaluation_10_test_8[0], "accuracy": evaluation_10_test_8[1]},
        "final_evaluation_8_9": {"loss": evaluation_10_test_9_10[0], "accuracy": evaluation_10_test_9_10[1]},
        "final_evaluation_full": {"loss": evaluation_10_test_full[0], "accuracy": evaluation_10_test_full[1]},
    }
    return results
