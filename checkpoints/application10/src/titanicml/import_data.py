"""Prepare environment for Titanic analysis
"""
import os
import yaml
import pandas as pd

config_file = os.path.join(os.path.dirname(__file__), "config.yaml")


def import_yaml_config(location: str = config_file) -> dict:
    """Wrapper to easily import YAML

    Args:
        location (str): File path

    Returns:
        dict: YAML content as dict
    """
    with open(location, "r", encoding="utf-8") as stream:
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

def create_variable_title(
    data: pd.DataFrame,
    variable_name: str = "Name"):
    """Transform name into title

    Args:
        data (pd.DataFrame): Dataset that should be modified
        variable_name (str, optional): Defaults to "Name".

    Returns:
        _type_: DataFrame with a title column
    """

    data["Title"] = (
        data[variable_name]
        .str.split(",").str[1]
        .str.split().str[0]
    )

    data.drop(
        labels=variable_name, axis=1, inplace=True
    )

    # Dona est présent dans le jeu de test à prédire mais
    # pas dans les variables d'apprentissage -> corrige
    data["Title"] = data["Title"].replace("Dona.", "Mrs.")

    return data
