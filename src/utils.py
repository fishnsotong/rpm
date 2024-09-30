import os
import pickle
import config
import numpy as np
import pandas as pd

def load_data():
    """
    Load the training, validation and test data from the specified filepaths.
    Parameters:
        None (defined globally in config.py)
    Returns:
        train_data (pd.DataFrame): The training data.
        val_data (pd.DataFrame): The validation data.
        test_data (pd.DataFrame): The test data.
    Raises:
        IOError: If the data cannot be loaded from the specified path.
    Example:
        train_data, val_data, test_data = load_data()
    """
    new_columns = ["Names", "Sequences", "Labels"]

    # not necessary, but DataFrames are easier to manipulate downstream
    train_data = pd.DataFrame(pd.read_pickle(config.TRAIN_DATA_PATH)).T
    val_data = pd.DataFrame(pd.read_pickle(config.VAL_DATA_PATH)).T
    test_data = pd.DataFrame(pd.read_pickle(config.TEST_DATA_PATH)).T

    train_data.columns = new_columns
    val_data.columns = new_columns
    test_data.columns = new_columns
    
    print("Data loaded successfully")

    return train_data, val_data, test_data

def save_model(model, model_name: str):
    """
    Save a machine learning model to a specified directory.
    Parameters:
        model: The machine learning model to be saved.
        model_name (str): The name of the file to save the model as.
    Returns:
        None
    Raises:
        IOError: If the model cannot be saved to the specified path.
    Example:
        save_model(my_model, 'model')
    """
    model_name = model_name + '.pkl'
    model_path = os.path.join(config.MODEL_SAVE_DIR, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model {model_name} saved at {model_path}")

def one_hot_encode(rna_str: str) -> np.ndarray:
    """
    Converts an RNA sequence string into a flattened one-hot encoded NumPy array.
    
    Each nucleotide in the RNA sequence is encoded as a 5-element vector:
    A -> [1, 0, 0, 0, 0] (adenine)
    C -> [0, 1, 0, 0, 0] (cytosine)
    G -> [0, 0, 1, 0, 0] (guanine)
    U -> [0, 0, 0, 1, 0] (uracil)
    X -> [0, 0, 0, 0, 1] (unknown or invalid nucleotide)

    Parameters:

    Returns:

    Example:

    """
    nucleotide_encoding = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'U': [0, 0, 0, 1, 0],
        'X': [0, 0, 0, 0, 1]
    }

    one_hot_encoded = []

    for nucleotide in rna_str:
        if nucleotide.upper() in nucleotide_encoding:
            one_hot_encoded.append(nucleotide_encoding[nucleotide])
        else:
            one_hot_encoded.append(nucleotide_encoding["X"])

    return np.array(one_hot_encoded).ravel()

def pad_encodings(max_length: int, encoded_sequences: list) -> np.ndarray:
    """
    Pads each sequence in a list of encoded sequences with zeros to ensure they all have the same length.

    Parameters:
    max_length (int): The length to which each sequence should be padded.
    encoded_sequences (list): A list of sequences, where each sequence is an array-like object.

    Returns:
    np.ndarray: A NumPy array containing the padded sequences.
    """

    # first we find the length of the longest sequence (we chose to do it outside bc we only need to do it once)
    # max_length = max(len(seq) for seq in encoded_sequences)
    
    padded_sequences = []
    for seq in encoded_sequences:
        pad_length = max_length - len(seq)
        # then we add zeros to each sequence until it reaches that max_length
        # do that by using np.zeros() until the difference is made up
        padded_seq = np.hstack([seq, np.zeros((pad_length))]) if pad_length > 0 else seq
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)             