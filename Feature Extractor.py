import os
import h5py
import numpy as np
import scipy.io as sio


def load_mat_file(file_path):
    """
    Load preprocessed .mat file containing epochsNormalised, fs, and triggers.
    """
    # Try loading using h5py for v7.3 or fallback to scipy for earlier versions
    try:
        with h5py.File(file_path, 'r') as f:
            epochs = np.array(f['epochsNormalized'])
            fs = f['fs'][()]  # Sampling frequency
            triggers = np.array(f['triggers']).flatten()  # Flatten triggers
        return epochs, fs, triggers
    except OSError:
        # Fallback for older MATLAB versions using scipy
        data = sio.loadmat(file_path)
        epochs = data['epochsNormalised']
        fs = data['fs'].item()
        triggers = data['triggers'].flatten()
        return epochs, fs, triggers


def extract_features(file_path):
    """
    Extract features from preprocessed EEG data.
    """
    # Load the preprocessed data
    epochs, fs, triggers = load_mat_file(file_path)

    # Initialize feature list (adjust this part according to the required features)
    features = []

    # Time-Domain Features: Mean, variance, and skewness per epoch and channel
    mean_feat = np.mean(epochs, axis=0)
    var_feat = np.var(epochs, axis=0)
    skewness_feat = np.apply_along_axis(lambda x: np.mean((x - np.mean(x)) ** 3), 0, epochs)

    # Combine all extracted time-domain features
    time_domain_features = np.concatenate([mean_feat, var_feat, skewness_feat], axis=1)

    # Add more feature extraction logic here as needed (e.g., frequency domain, ERP, etc.)

    # Return extracted features
    return time_domain_features


def save_features(features, folder_name, file_name):
    """
    Save extracted features to a .npy file for future use.
    Creates the directory if it doesn't exist.
    """
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save the features to a .npy file
    output_file = f'{folder_name}/{file_name}_features.npy'
    np.save(output_file, features)
    print(f'Features saved to {output_file}')


# Usage Example:
if __name__ == "__main__":
    file_path = r'G:\locked in\Preprocessed_data\Preprocessed_P1_low1.mat'
    features = extract_features(file_path)
    save_features(features, 'G:/locked in/Extracted_Features', 'P1_low1')