"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import sys
import os

from sklearn.metrics import confusion_matrix
import src.data.import_data as imp
import src.features.build_features as bf
import src.models.train_evaluate as te


# PARAMETRES -------------------------------

config = imp.import_yaml_config("configuration/config.yaml")
path_secrets_yaml = "configuration/secrets.yaml"
if os.path.exists(path_secrets_yaml):
    secrets = imp.import_yaml_config(path_secrets_yaml)
    API_TOKEN = secrets["api"]["token"]

# Number trees as command line argument
N_TREES = int(sys.argv[1]) if len(sys.argv) == 2 else 20


LOCATION_TRAIN = config["path"]["train"]
LOCATION_TEST = config["path"]["test"]
TEST_FRACTION = config["model"]["test_fraction"]


# FEATURE ENGINEERING --------------------------------

TrainingData = imp.import_data(LOCATION_TRAIN)
TestData = imp.import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = imp.create_variable_title(TrainingData)
TestData = imp.create_variable_title(TestData)


## IMPUTATION DES VARIABLES ================


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = bf.check_has_cabin(TrainingData)
TestData = bf.check_has_cabin(TestData)

TrainingData = bf.ticket_length(TrainingData)
TestData = bf.ticket_length(TestData)


train, test = te.split_train_test_titanic(
    TrainingData,
    fraction_test=TEST_FRACTION
    )


# MODELISATION: RANDOM FOREST ----------------------------

pipe = te.random_forest_titanic(
    data=TrainingData, fraction_test=TEST_FRACTION, n_trees=N_TREES
)

pipe.fit(train.drop("Survived", axis="columns"), train["Survived"])


# EVALUATE ----------------------------

rdmf_score = pipe.score(test.drop("Survived", axis="columns"), test["Survived"])
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
