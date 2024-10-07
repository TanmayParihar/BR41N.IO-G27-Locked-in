import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping


# Load features from Extracted_Features folder
def load_features(folder_path):
    features = []
    labels = []

    # Mapping of filenames to classes (adjust based on your data structure)
    label_mapping = {
        'P1_high1': 1, 'P1_high2': 1,  # High performance class
        'P1_low1': 0, 'P1_low2': 0,  # Low performance class
        'P2_high1': 1, 'P2_high2': 1,  # High performance class
        'P2_low1': 0, 'P2_low2': 0  # Low performance class
    }

    # Loop through all feature files and load them
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            # Load feature file
            feature_data = np.load(os.path.join(folder_path, file_name))

            # Extract label from the file name
            base_name = file_name.replace('_features.npy', '')
            label = label_mapping.get(base_name, None)

            if label is not None:
                features.append(feature_data)
                labels.append(label)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


# Prepare the data (split into training and testing sets, standardize features)
def prepare_data(features, labels):
    # Split the data into training and test sets
    X_train, X_test, y_train_int, y_test_int = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Standardize the features (scaling the data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # One-hot encode the labels for binary classification
    y_train = to_categorical(y_train_int, num_classes=2)
    y_test = to_categorical(y_test_int, num_classes=2)

    return X_train, X_test, y_train, y_test, y_train_int, y_test_int


# Build the CNN-RNN hybrid model
def build_hybrid_model(input_shape):
    model = Sequential()

    # CNN for spatial feature extraction
    model.add(
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Increased dropout for better generalization

    # CNN layer 2
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))  # Increased dropout for better generalization

    # LSTM for temporal pattern recognition
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))  # Increased dropout to prevent overfitting

    # Fully connected layers with L2 regularization
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))  # Increased dropout
    model.add(Dense(2, activation='softmax'))  # Output for 2 classes

    # Compile the model with lower learning rate and Adam optimizer
    optimizer = Adam(learning_rate=0.00001)  # Reduced learning rate for better convergence
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Cross-validation training with early stopping and class weighting
def cross_validation_train(X_train, y_train, y_train_int):
    # Check class distribution
    unique, counts = np.unique(y_train_int, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Adjust n_splits based on the smallest class size
    smallest_class_size = min(counts)
    n_splits = min(5, smallest_class_size)

    # Setup k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Calculate class weights to handle class imbalance using integer labels
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    class_weights_dict = dict(enumerate(class_weights))

    best_accuracy = 0
    best_model = None

    # Loop through the k-fold splits
    for train_index, val_index in skf.split(X_train, y_train_int):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_hybrid_model(input_shape)

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, validation_data=(X_val_fold, y_val_fold),
                  class_weight=class_weights_dict, callbacks=[early_stopping])

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold)

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model

    return best_model, best_accuracy


# Main program
if __name__ == "__main__":
    # Folder path for extracted features
    folder_path = r'G:\locked in\Extracted_Features'

    # Step 1: Load features and labels
    features, labels = load_features(folder_path)

    # Step 2: Prepare the data
    X_train, X_test, y_train, y_test, y_train_int, y_test_int = prepare_data(features, labels)

    # Step 3: Perform cross-validation and select the best model
    best_model, best_accuracy = cross_validation_train(X_train, y_train, y_train_int)
    print(f"Best cross-validation accuracy: {best_accuracy:.2f}")

    # Step 4: Evaluate the best model on the test set
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")
