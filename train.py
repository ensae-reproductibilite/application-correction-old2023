"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import argparse
from sklearn.metrics import confusion_matrix
from joblib import dump

import src.data.import_data as imp
import src.features.build_features as bf
import src.models.train_evaluate as te


# PARAMETRES -------------------------------

# Paramètres ligne de commande
parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()

# Paramètres YAML
config = imp.import_yaml_config("configuration/config.yaml")
base_url = "https://minio.lab.sspcloud.fr/projet-formation/ensae-reproductibilite/data/raw"
API_TOKEN = config.get("jeton_api")
LOCATION_TRAIN = config.get("train_path", f"{base_url}/train.csv")
LOCATION_TEST = config.get("test_path", f"{base_url}/test.csv")
TEST_FRACTION = config.get("test_fraction", .1)
N_TREES = args.n_trees


# FEATURE ENGINEERING --------------------------------

TrainingData = imp.import_data(LOCATION_TRAIN)
TestData = imp.import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = bf.create_variable_title(TrainingData)
TestData = bf.create_variable_title(TestData)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = bf.check_has_cabin(TrainingData)
TestData = bf.check_has_cabin(TestData)

TrainingData = bf.ticket_length(TrainingData)
TestData = bf.ticket_length(TestData)

train, test = te.split_train_test_titanic(
    TrainingData,
    fraction_test=TEST_FRACTION
)
X_train, y_train = train.drop("Survived", axis="columns"), train["Survived"]
X_test, y_test = test.drop("Survived", axis="columns"), test["Survived"]


# MODELISATION: RANDOM FOREST ----------------------------

pipe = te.build_pipeline(n_trees=N_TREES)

pipe.fit(X_train, y_train)


# EVALUATE ----------------------------

rdmf_score = pipe.score(X_test, y_test)
print(
    f"{round(rdmf_score * 100)} % de bonnes réponses sur les données de test pour validation \
            (résultat qu'on attendrait si on soumettait notre prédiction \
                sur le dataset de test.csv)"
)

print("matrice de confusion")
print(
    confusion_matrix(
        test["Survived"], pipe.predict(test.drop("Survived", axis="columns"))
    )
)

dump(pipe, "model.joblib")