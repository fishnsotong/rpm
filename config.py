# global variables for the training pipeline

# filepaths to processed data
TRAIN_DATA_PATH = "data/processed/train_data.pkl"
VAL_DATA_PATH = "data/processed/val_data.pkl"
TEST_DATA_PATH = "data/processed/test_data.pkl"


# directory to save models in
MODEL_SAVE_DIR = "models"

# hyperparameters for different models (replace this with optimal parameters)

HYPERPARAMS = {
    "logreg": {
        "C": 1.0,
        "max_iter": 1000
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    }
}

# hyperparameters for grid search

HYPERPARAMS_GRID = {
}
