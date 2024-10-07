import numpy as np
import os
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Load raw data
def load_raw_data(folder):
    X_raw = []
    y_raw = []
    for file in os.listdir(folder):
        if file.endswith(".mat"):
            data = sio.loadmat(os.path.join(folder, file))
            trig = data['trig'].flatten()
            eeg_data = data['y']
            X_raw.append(eeg_data)
            y_raw.append(trig)
    X_raw = np.vstack(X_raw)
    y_raw = np.hstack(y_raw)
    return X_raw, y_raw


# Preprocess raw data
def preprocess_raw_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# Transform labels to binary classification
def transform_labels(y):
    return np.where(y > 0, 1, 0)  # 1 for target and 0 for non-target


# Reshape data to have time steps (e.g., 3 time steps)
def reshape_data(X, y, time_steps=3):
    num_samples = (X.shape[0] // time_steps) * time_steps  # Adjust to be divisible by time_steps
    X_reshaped = X[:num_samples].reshape(-1, time_steps, X.shape[1])
    y_reshaped = y[:num_samples:time_steps]  # Downsample labels to match reshaped X
    return X_reshaped, y_reshaped


# Build the CNN-RNN hybrid model
def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Cross-validation training
def cross_validation_train(X, y):
    skf = StratifiedKFold(n_splits=5)
    best_accuracy = 0
    best_model = None
    for train_index, val_index in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        input_shape = (X_train_fold.shape[1], X_train_fold.shape[2])
        model = build_model(input_shape)
        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=4, validation_data=(X_val_fold, y_val_fold))

        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model

    return best_model, best_accuracy


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.2f}")


# Main script
if __name__ == "__main__":
    # Load and preprocess raw data
    mat_folder = r'G:\locked in\Data'
    X_raw, y_raw = load_raw_data(mat_folder)
    X_raw_scaled = preprocess_raw_data(X_raw)

    # Transform labels
    y_raw_transformed = transform_labels(y_raw)

    # Reshape data to include time steps
    X_raw_reshaped, y_raw_reshaped = reshape_data(X_raw_scaled, y_raw_transformed, time_steps=3)

    # Cross-validation training
    best_model, best_accuracy = cross_validation_train(X_raw_reshaped, y_raw_reshaped)
    print(f"Best cross-validation accuracy: {best_accuracy:.2f}")

    # Test the model on the same dataset (for now)
    evaluate_model(best_model, X_raw_reshaped, y_raw_reshaped)
