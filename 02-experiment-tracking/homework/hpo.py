import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("num_trials", num_trials)
            mlflow.log_param("data_path", data_path)
            mlflow.log_metric("num_samples_train", len(y_train))
            mlflow.log_metric("num_samples_val", len(y_val))
            mlflow.log_metric("num_features", X_train.shape[1])
            mlflow.log_metric("num_trials", num_trials)
            mlflow.set_tag("model", "RandomForestRegressor")
            mlflow.set_tag("optimizer", "Hyperopt")
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("version", "0.1.0")
            mlflow.set_tag("data_source", "NYC Taxi Trip Data")

            mlflow.set_tag("run_id", mlflow.active_run().info.run_id)
            mlflow.set_tag("tracking_uri", mlflow.get_tracking_uri())
            mlflow.set_tag("created_by", "Marina Galchenkova")
            mlflow.set_tag("created_on", "2023-10-01")
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(rf, "random-forest-model")
        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    run_optimization()