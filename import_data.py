import os
import yaml
import pandas as pd

def import_yaml_config(filename: str = "toto.yaml") -> dict:
    dict_config = {}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as stream:
            dict_config = yaml.safe_load(stream)
    return dict_config


def import_data(path: str) -> pd.DataFrame :
    """Import Titanic datasets
    Args:
        path (str): File location
    Returns:
        pd.DataFrame: Titanic dataset
    """

    data = pd.read_csv(path)
    data = data.drop(columns="PassengerId")

    return data
