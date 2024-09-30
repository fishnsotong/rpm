import os
import pickle
import config
import pandas as pd

def load_data():
    """
    Load the training, validation and test data from the specified filepaths.
    Args:
        None
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
    train_data = pd.DataFrame(pd.read_pickle(config.TRAIN_DATA_PATH)).T; train_data.columns = new_columns
    val_data = pd.DataFrame(pd.read_pickle(config.VAL_DATA_PATH)).T; val_data.columns = new_columns
    test_data = pd.DataFrame(pd.read_pickle(config.TEST_DATA_PATH)).T; test_data.columns = new_columns

    print("Data loaded successfully")

    return train_data, val_data, test_data

def save_model(model, model_name: str):
    """
    Save a machine learning model to a specified directory.
    Args:
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