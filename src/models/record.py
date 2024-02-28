import mlflow
from sklearn.metrics import f1_score, accuracy_score

def log_rf_to_mlflow(
    pipe,
    X_test,
    y_test,
    n_trees: int = 20,
    max_depth=None,
    max_features="sqrt",
    application_number="appli21",
):
    """Log a scikit-learn trained GridSearchCV object as an MLflow experiment."""
    # Set up MLFlow context
    mlflow.set_experiment(experiment_name="titanicml")

    run_name = f"run {n_trees} - {max_depth} - {max_features}"
    parameters = {
        "n_trees": n_trees,
        "max_depth": max_depth,
        "max_features": max_features,
    }
    # Compute metrics
    f1 = f1_score(y_test, pipe.predict(X_test))
    accuracy = accuracy_score(y_test, pipe.predict(X_test))
    
    with mlflow.start_run(run_name=run_name):
        # Log each parameter individually
        for key, value in parameters.items():
            mlflow.log_param(key, value)
        
        # Log each metric individually
        mlflow.log_metric("mean_test_f1", f1)
        mlflow.log_metric("mean_accuracy_f1", accuracy)
        
        mlflow.log_param("application_number", application_number)
        
        # Log the model
        mlflow.sklearn.log_model(pipe, "model")
