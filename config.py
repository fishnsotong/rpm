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
     "logreg": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ['l1', 'l2'],
        "max_iter": [100, 200, 500, 1000],
        "solver": ["liblinear"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],    # number of trees in the forest
        "max_depth": [None, 10, 20, 30],        # maximum depth of each tree
        "min_samples_leaf": [1, 2, 4],          # minimum number of samples required to be at a leaf node
        "min_samples_split": [2, 5, 10],        # minimum number of samples required to split an internal node
        "bootstrap": [True, False],             # method of selecting samples for training each tree
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.001, 0.01, 0.1],
        "gamma": [0, 0.1, 0.2],
        "reg_lambda": [1, 10, 100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "random_state": 42
    }
}
