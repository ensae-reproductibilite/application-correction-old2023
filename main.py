"""
Prediction de la survie d'un individu sur le Titanic
"""

# GESTION ENVIRONNEMENT --------------------------------

import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import import_data as imp
import build_features as bf
import train_evaluate as te



# PARAMETRES -------------------------------

# Paramètres ligne de commande
parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()

# Paramètres YAML
config = imp.import_yaml_config("config.yaml")
API_TOKEN = config.get("jeton_api")
LOCATION_TRAIN = config.get("train_path", "train.csv")
LOCATION_TEST = config.get("test_path", "test.csv")
TEST_FRACTION = config.get("test_fraction", .1)
N_TREES = args.n_trees


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