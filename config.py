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
        "max_iter": 1000,
        "penalty": "l2",
        "solver": "liblinear",
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "bootstrap": False,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 10,
        "learning_rate": 0.1,
        "gamma": 0.1,
        "reg_lambda": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
}

# hyperparameters for grid search

HYPERPARAMS_GRID = {
}
