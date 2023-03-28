"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import sys
import os

import titanicml.import_data as imp
import titanicml.build_features as bf
import titanicml.train_evaluate as te


# PARAMETRES -------------------------------


config = imp.import_yaml_config()
path_secrets_yaml = "configuration/secrets.yaml"

if os.path.exists(path_secrets_yaml):
    secrets = imp.import_yaml_config(path_secrets_yaml)
    API_TOKEN = secrets["api"]['token']

# Number trees as command line argument
N_TREES = int(sys.argv[1]) if len(sys.argv) == 2 else 20


LOCATION_TRAIN = config['path']['train']
LOCATION_TEST = config['path']['test']
TEST_FRACTION = config['model']['test_fraction']



# FEATURE ENGINEERING --------------------------------

TrainingData = imp.import_data(LOCATION_TRAIN)
TestData = imp.import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = imp.create_variable_title(TrainingData)
TestData = imp.create_variable_title(TestData)


## IMPUTATION DES VARIABLES ================


TrainingData = bf.fill_na_titanic(TrainingData)
TestData = bf.fill_na_titanic(TestData)

TrainingData = bf.label_encoder_titanic(TrainingData)
TestData = bf.label_encoder_titanic(TestData)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = bf.check_has_cabin(TrainingData)
TestData = bf.check_has_cabin(TestData)

TrainingData = bf.ticket_length(TrainingData)
TestData = bf.ticket_length(TestData)



# MODELISATION: RANDOM FOREST ----------------------------

model = te.random_forest_titanic(
    data=TrainingData,
    fraction_test=TEST_FRACTION,
    n_trees=N_TREES
)
