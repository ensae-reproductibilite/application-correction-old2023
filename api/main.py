"""A simple API to expose our trained RandomForest model for Tutanic survival."""
import requests
from fastapi import FastAPI
from joblib import load

import pandas as pd
import mlflow

# GET PRODUCTION MODEL -------------


model_name = "titanic"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


# USE PRODUCTION MODEL IN APP ----------

app = FastAPI(
    title="Prédiction de survie sur le Titanic",
    description=
    "<b>Application de prédiction de survie sur le Titanic</b> 🚢 <br>Une version par API pour faciliter la réutilisation du modèle 🚀" +\
        "<br><br><img src=\"https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg\" width=\"200\">" +\
        "<br>Version avec le modèle de prod"
    )


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API de prédiction de survie sur le Titanic",
        "Model_name": 'Titanic ML',
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    pclass: int = 3,
    sex: str = "female",
    age: float = 29.0,
    sib_sp: int = 1,
    parch: int = 1,
    fare: float = 16.5,
    embarked: str = "S",
    has_cabin: int = 1,
    ticket_len: int = 7
) -> str:
    """
    """

    df = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sib_sp],
            "parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked],
            "hasCabin": [has_cabin],
            "Ticket_Len": [ticket_len] 
        }
    )

    prediction = "Survived 🎉" if int(model.predict(df)) == 1 else "Dead ⚰️"

    return prediction