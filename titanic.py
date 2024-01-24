"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()


# FONCTIONS --------------------

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


def split_train_test_titanic(
    data: pd.DataFrame,
    y_index: int = 0,
    fraction_test: float = 0.1):
    """Split Titanic dataset in train and test sets
    Args:
        data (pd.DataFrame): Titanic dataset
        y_index (int, optional): Positional index for target variable.
        fraction_test (float, optional):
            Fraction of observation dedicated to test dataset.
            Defaults to 0.1.
    Returns:
        Four elements : X_train, X_test, y_train, y_test
    """

    y = data.iloc[:, y_index].values
    X = data.iloc[:, 1:12].values

    # Feature Scaling
    scaler_x = MinMaxScaler((-1, 1))
    X = scaler_x.fit_transform(X)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=fraction_test)

    return X_train, X_test, y_train, y_test

def random_forest_titanic(
    data: pd.DataFrame,
    fraction_test: float = 0.9,
    n_trees: int = 20):
    """Random forest model for Titanic survival
    Args:
        data (pd.DataFrame): _description_
        fraction_test (float, optional): _description_. Defaults to 0.9.
        n_trees (int, optional): _description_. Defaults to 20.
    Returns:
        _type_: _description_
    """

    X_train, X_test, y_train, y_test = \
    split_train_test_titanic(
        data,
        fraction_test=fraction_test
    )

    rdmf = RandomForestClassifier(n_estimators=n_trees)
    rdmf.fit(X_train, y_train)

    # calculons le score sur le dataset d'apprentissage et sur le dataset de test
    # (10% du dataset d'apprentissage mis de côté)
    # le score étant le nombre de bonne prédiction
    rdmf_score = rdmf.score(X_test, y_test)
    print(
        f"{round(rdmf_score * 100)} % de bonnes réponses sur les données de test pour validation \
            (résultat qu'on attendrait si on soumettait notre prédiction \
                sur le dataset de test.csv)"
    )

    print("matrice de confusion")
    confusion_matrix(y_test, rdmf.predict(X_test))

    return rdmf, X_train, X_test, y_train, y_test


# PARAMETRES -------------------------------

config = import_yaml_config("config.yaml")

API_TOKEN = config.get("jeton_api")
LOCATION_TRAIN = config.get("train_path", "train.csv")
LOCATION_TEST = config.get("test_path", "test.csv")
TEST_FRACTION = config.get("test_fraction", .1)
N_TREES = args.n_trees


# FEATURE ENGINEERING --------------------------------

TrainingData = import_data(LOCATION_TRAIN)
TestData = import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = create_variable_title(TrainingData)
TestData = create_variable_title(TestData)


## IMPUTATION DES VARIABLES ================


TrainingData = fill_na_titanic(TrainingData)
TestData = fill_na_titanic(TestData)

TrainingData = label_encoder_titanic(TrainingData)
TestData = label_encoder_titanic(TestData)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = check_has_cabin(TrainingData)
TestData = check_has_cabin(TestData)

TrainingData = ticket_length(TrainingData)
TestData = ticket_length(TestData)



# MODELISATION: RANDOM FOREST ----------------------------

model = random_forest_titanic(
    data=TrainingData,
    fraction_test=TEST_FRACTION,
    n_trees=N_TREES
)