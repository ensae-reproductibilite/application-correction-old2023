import pandas as pd


def create_variable_title(data: pd.DataFrame, variable_name: str = "Name"):
    """Transform name into title
    Args:
        data (pd.DataFrame): Dataset that should be modified
        variable_name (str, optional): Defaults to "Name".
    Returns:
        _type_: DataFrame with a title column
    """

    data["Title"] = data[variable_name].str.split(",").str[1].str.split().str[0]

    data.drop(labels=variable_name, axis=1, inplace=True)

    # Dona est présent dans le jeu de test à prédire mais
    # pas dans les variables d'apprentissage -> corrige
    data["Title"] = data["Title"].replace("Dona.", "Mrs.")

    return data


def check_has_cabin(data: pd.DataFrame) -> pd.DataFrame:
    """Label if observation has a cabin
    Args:
        data (pd.DataFrame): Titanic dataset
    Returns:
        pd.DataFrame : Titanic dataset with a new observation
    """
    data["hasCabin"] = data.Cabin.notnull().astype(int)
    data = data.drop(labels="Cabin", axis=1)
    return data


def ticket_length(data: pd.DataFrame) -> pd.DataFrame:
    """Label observation ticket length
    Args:
        data (pd.DataFrame): Titanic dataset
    Returns:
        pd.DataFrame: Titanic dataset with a new ticket variable
    """
    data["Ticket_Len"] = data["Ticket"].str.len()
    data = data.drop(labels="Ticket", axis=1)
    return data
