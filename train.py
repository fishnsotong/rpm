import os
import pandas as pd
import argparse

# model imports
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# evaluation imports
from sklearn.metrics import accuracy_score, classification_report

# utility imports
import config
from src.utils import load_data, save_model, one_hot_encode, pad_encodings

def get_model(model_name: str, tune: bool):
    """
    Initialize a machine learning model based on the model name.

    Parameters:
        model_name (str): The name of the model to initialize.
    """
    if tune == True:
        if model_name == "logreg":
            return LogisticRegression(), "logistic regression"
        elif model_name == "random_forest":
            return RandomForestClassifier(n_jobs=-1), "random forest"
        elif model_name == "xgboost":
            return XGBClassifier(eval_metric='logloss', n_jobs=-1), "xgboost"
        else:
            raise ValueError(f'Model {model_name} is not supported')
    else:
        if model_name == "logreg":
            return LogisticRegression(**config.HYPERPARAMS["logreg"]), "logistic regression"
        elif model_name == "random_forest":
            return RandomForestClassifier(**config.HYPERPARAMS["random_forest"]), "random forest"
        elif model_name == "xgboost":
            return XGBClassifier(**config.HYPERPARAMS["xgboost"]), "xgboost"
        else:
            raise ValueError(f'Model {model_name} is not supported')
    
def train(model_name: str, tune: bool):

    # step 1: load and prepare the data (training, validation, test)
    # including one-hot encoding the sequences and padding them
    train_data, val_data, test_data = load_data()
    N = max(len(x) for x in pd.concat([train_data['Sequences'], val_data['Sequences'], test_data['Sequences']]).tolist()) * 5

    X_train = pad_encodings(N, [one_hot_encode(seq) for seq in train_data['Sequences']])
    X_val = pad_encodings(N, [one_hot_encode(seq) for seq in val_data['Sequences']])
    X_test = pad_encodings(N, [one_hot_encode(seq) for seq in test_data['Sequences']])

    # convert labels to numpy arrays
    y_train = train_data['Labels'].astype(int).to_numpy()
    y_val = val_data['Labels'].astype(int).to_numpy()
    y_test = test_data['Labels'].astype(int).to_numpy()

    # step 2: initialize and train the model
    model, name = get_model(model_name, tune)

    if tune == True:
        # Use GridSearchCV to tune hyperparameters
        print("\nTraining model: ", name.upper(), "...\n")
        grid_search = GridSearchCV(estimator=model, param_grid=config.HYPERPARAMS_GRID[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # print the best hyperparameters found by GridSearchCV
        print("Best hyperparameters found by GridSearchCV:", grid_search.best_params_)
        print("Best accuracy score found by GridSearchCV:", grid_search.best_score_)    

        # step 3: train the best model using the training data
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # step 4: evaluate the best model on the validation set
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation accuracy: {val_accuracy:.3f}")
        print("Validation Classification Report:")
        print(classification_report(y_val, y_val_pred))

        # step 5: evaluate the best model on the test set
        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test accuracy: {test_accuracy:.3f}")
        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred))

        # step 6: save the best model
        save_model(best_model, model_name)
    
    else:
        # TODO: why do we need a validation set if we use k-fold cross-validation?

        # fit the model to the training data
        print("\nTraining model: ", name.upper(), "...\n")
        model.fit(X_train, y_train)

        # step 3: evaluate the model on the validation set
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation accuracy: {val_accuracy:.3f}")
        print("Validation Classification Report:")
        print(classification_report(y_val, y_val_pred))

        # step 4: evaluate the model on the test set
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test accuracy: {test_accuracy:.3f}")
        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred))

        # step 5: save the model
        save_model(model, model_name)
    
def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, required=True, 
                        choices=["logreg", "random_forest", "xgboost"], 
                        help='Model to train: logreg, random_forest, xgboost')
    parser.add_argument('--tune', action='store_true', 
                        help='Flag to indicate if hyperparameter tuning is required')
    args = parser.parse_args()

    train(args.model, args.tune)

if __name__ == '__main__':
    main()