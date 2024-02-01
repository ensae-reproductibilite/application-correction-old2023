"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import argparse
import titanicml as tml


# PARAMETRES -------------------------------

# Paramètres ligne de commande
parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()

# Paramètres YAML
config = tml.import_yaml_config("configuration/config.yaml")
base_url = "https://minio.lab.sspcloud.fr/projet-formation/ensae-reproductibilite/data/raw"
API_TOKEN = config.get("jeton_api")
LOCATION_TRAIN = config.get("train_path", f"{base_url}/train.csv")
LOCATION_TEST = config.get("test_path", f"{base_url}/test.csv")
TEST_FRACTION = config.get("test_fraction", .1)
N_TREES = args.n_trees


# FEATURE ENGINEERING --------------------------------

TrainingData = tml.import_data(LOCATION_TRAIN)
TestData = tml.import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = tml.create_variable_title(TrainingData)
TestData = tml.create_variable_title(TestData)


## IMPUTATION DES VARIABLES ================

TrainingData = tml.fill_na_titanic(TrainingData)
TestData = tml.fill_na_titanic(TestData)

TrainingData = tml.label_encoder_titanic(TrainingData)
TestData = tml.label_encoder_titanic(TestData)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = tml.check_has_cabin(TrainingData)
TestData = tml.check_has_cabin(TestData)

TrainingData = tml.ticket_length(TrainingData)
TestData = tml.ticket_length(TestData)


# MODELISATION: RANDOM FOREST ----------------------------

model = tml.random_forest_titanic(
    data=TrainingData,
    fraction_test=TEST_FRACTION,
    n_trees=N_TREES
)