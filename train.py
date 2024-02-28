"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

from pathlib import Path
import argparse
from joblib import dump
from sklearn.model_selection import GridSearchCV

import src.data.import_data as imp
import src.features.build_features as bf
import src.models.log as mlog
import src.models.train_evaluate as te
import src.models.record as smr

# PARAMETRES -------------------------------

# Paramètres ligne de commande
parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbres")
parser.add_argument("--appli", type=str, default="appli21", help="Application number")
parser.add_argument("--max_depth", type=int, default=None, help="The maximum depth of the tree.")
parser.add_argument("--max_features", type=str, default="sqrt", help="The number of features to consider when looking for the best split")
args = parser.parse_args()

# Paramètres YAML
config = imp.import_yaml_config("configuration/config.yaml")
base_url = (
    "https://minio.lab.sspcloud.fr/projet-formation/ensae-reproductibilite/data/raw"
)
API_TOKEN = config.get("jeton_api")
LOCATION_TRAIN = config.get("train_path", f"{base_url}/train.csv")
LOCATION_TEST = config.get("test_path", f"{base_url}/test.csv")
TEST_FRACTION = config.get("test_fraction", 0.1)
N_TREES = args.n_trees
APPLI_ID = args.appli
EXPERIMENT_NAME = "titanicml"

# FEATURE ENGINEERING --------------------------------

titanic_raw = imp.import_data(LOCATION_TRAIN)

# Create a 'Title' variable
titanic_intermediate = bf.feature_engineering(titanic_raw)


train, test = te.split_train_test_titanic(
    titanic_intermediate, fraction_test=TEST_FRACTION
)
X_train, y_train = train.drop("Survived", axis="columns"), train["Survived"]
X_test, y_test = test.drop("Survived", axis="columns"), test["Survived"]


def log_local_data(data, filename):
    data.to_csv(f"data/intermediate/{filename}.csv", index=False)


output_dir = Path("data/intermediate")
output_dir.mkdir(parents=True, exist_ok=True)

log_local_data(X_train, "X_train")
log_local_data(X_test, "X_test")
log_local_data(y_train, "y_train")
log_local_data(y_test, "y_test")


# MODELISATION: RANDOM FOREST ----------------------------

pipe = te.build_pipeline(
    n_trees=N_TREES, categorical_features=["Embarked", "Sex"],
    max_depth=args.max_depth, max_features=args.max_features
)

pipe.fit(X_train, y_train)

smr.log_rf_to_mlflow(
    pipe=pipe, X_test=X_test, y_test=y_test,
    application_number = args.appli
)

