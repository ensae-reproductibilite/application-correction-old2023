import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def split_train_test_titanic(data: pd.DataFrame, fraction_test: float = 0.1):
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

    train = data.sample(frac=1 - fraction_test, random_state=435)
    test = data.drop(train.index)

    return train, test


def build_pipeline(
    n_trees: int = 20,
    numeric_features=["Age", "Fare"],
    categorical_features=["Title", "Embarked", "Sex"],
):
    """Random forest model for Titanic survival

    Args:
        n_trees (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=n_trees)),
        ]
    )

    return pipe
