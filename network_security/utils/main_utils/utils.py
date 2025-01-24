import yaml
import os
import sys
import dill
import pickle
import numpy as np
from network_security.exceptions.exception import NetworkSecurityException
from network_security.logging.logger import logging


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
