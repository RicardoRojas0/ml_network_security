import yaml
import os
import sys
import dill
import pickle
import numpy as np
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


# Read YAML file
def read_yaml_file(file_path: str) -> dict:
    """
    Read the YAML file and return the data as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The data from the YAML file as a dictionary.
    """
    try:
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# Write YAML file
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# Saving a numpy array
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Saving the pickle file
def save_object(file_path: str, object: object):
    try:
        logging.info("Entered the save_object method of main utils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Load the pickle file
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# Load the numpy array
def load_numpy_array_data(file_path: str) -> np.array:
    """
    Args:
        file_path (str): The path to the file where the numpy array will be saved.

    Raises:
        NetworkSecurityException: If there is an error during the saving process.

    Returns:
        np.array: The numpy array to be saved.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        # Dictionary to save the report
        report = {}

        # Looping to each of the models and parameters within dictionaries
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # Brute forcing each parameter to find the best fit
            grid_search = GridSearchCV(model, param, cv=3)
            grid_search.fit(X_train, y_train)

            # Setting the best parameters to the model
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # Making prediction for train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Obtaining the R2 Score of each of the sets
            train_model_r2_score = r2_score(y_true=y_train, y_pred=y_train_pred)
            test_model_r2_score = r2_score(y_true=y_test, y_pred=y_test_pred)

            # Getting the information into the report dictionary
            report[list(models.keys())[i]] = test_model_r2_score

            return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)
