import pandas as pd
from sklearn.preprocessing import LabelEncoder

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


def fill_na_column(
    data: pd.DataFrame,
    column: str = "Age",
    value: float = 0.0
) -> pd.DataFrame :
    """Imputation for a given column
    Args:
        data (pd.DataFrame): Dataset that should be modified
        column (str, optional): Column that should be imputed. Defaults to "Age".
    Returns:
        pd.DataFrame: Initial dataset with mean-imputed column
    """
    data[column] = data[column].fillna(value)
    return data


def fill_na_titanic(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Pipeline of imputations
    Args:
        data (pd.DataFrame): Titanic dataframe
    Returns:
        pd.DataFrame: Titanic dataframe with age, embarked and fare columns imputed
    """

    # Age variable imputation
    mean_age_training = data["Age"].mean().round()
    data = fill_na_column(data, "Age", mean_age_training)

    # Embarked imputed to S
    data = fill_na_column(data, "Embarked", "S")

    # Mean Fare imputation
    mean_fare_imputation = data["Fare"].mean()
    data = fill_na_column(data, "Fare", mean_fare_imputation)

    return data


def label_encoder_titanic_column(
    data: pd.DataFrame, column: str = "Sex"
) -> pd.DataFrame:
    """Label encoder for a given column
    Args:
        data (pd.DataFrame): Titanic dataset
        column (str, optional): Column that should be encoded. Defaults to "Sex".
    Returns:
        pd.DataFrame: Titanic with column encoded
    """
    label_encoder_column = LabelEncoder()
    data[column] = \
        label_encoder_column.fit_transform(data[column].values)

    return data


def label_encoder_titanic(data: pd.DataFrame) -> pd.DataFrame:
    """Label encoding pipeline for Titanic
    Args:
        data (pd.DataFrame): Titanic dataset
    Returns:
        pd.DataFrame: Titanic with Sex, Title and Embarked columns encoded
    """

    data = label_encoder_titanic_column(data, "Sex")
    data = label_encoder_titanic_column(data, "Title")
    data = label_encoder_titanic_column(data, "Embarked")

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
