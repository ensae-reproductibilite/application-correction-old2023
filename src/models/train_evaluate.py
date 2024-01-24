import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def split_train_test_titanic(
    data: pd.DataFrame, y_index: int = 0, fraction_test: float = 0.1
):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=fraction_test)

    return X_train, X_test, y_train, y_test


def random_forest_titanic(
    data: pd.DataFrame, fraction_test: float = 0.9, n_trees: int = 20
):
    """Random forest model for Titanic survival
    Args:
        data (pd.DataFrame): _description_
        fraction_test (float, optional): _description_. Defaults to 0.9.
        n_trees (int, optional): _description_. Defaults to 20.
    Returns:
        _type_: _description_
    """

    X_train, X_test, y_train, y_test = split_train_test_titanic(
        data, fraction_test=fraction_test
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
