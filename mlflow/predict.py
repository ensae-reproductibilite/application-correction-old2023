import mlflow
import pandas as pd


model_name = "titanic"
model_version = 1

loaded_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
    )


def create_data(
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

    return df


data = pd.concat([
    create_data(),
    create_data(sex="male")
])

print(
    loaded_model.predict(pd.DataFrame(data))
)