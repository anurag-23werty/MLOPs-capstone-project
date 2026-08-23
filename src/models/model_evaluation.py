import numpy as np
import pandas as pd
import pickle
import json
import logging
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn
import dagshub

from src.logger import logging
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

url = os.getenv("TRACKING_URI")


# ---------------------------------------------------------
# Configure MLflow + DagsHub
# ---------------------------------------------------------

mlflow.set_tracking_uri(url)

dagshub.init(
    repo_owner="anurag-23werty",
    repo_name="MLOPs-capstone-project",
    mlflow=True
)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def load_model(file_path: str):
    """Load the trained model from a pickle file."""

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logging.info("Model loaded from %s", file_path)

        return model

    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise

    except Exception as e:
        logging.error(
            "Unexpected error occurred while loading the model: %s",
            e
        )
        raise


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    try:
        df = pd.read_csv(file_path)

        logging.info("Data loaded from %s", file_path)

        return df

    except pd.errors.ParserError as e:
        logging.error(
            "Failed to parse the CSV file: %s",
            e
        )
        raise

    except Exception as e:
        logging.error(
            "Unexpected error occurred while loading the data: %s",
            e
        )
        raise


# ---------------------------------------------------------
# Evaluate Model
# ---------------------------------------------------------

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the model and return evaluation metrics."""

    try:

        # Predictions
        y_pred = clf.predict(X_test)

        # Prediction probabilities
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_pred_proba
        )

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

        logging.info(
            "Model evaluation metrics calculated"
        )

        return metrics_dict

    except Exception as e:

        logging.error(
            "Error during model evaluation: %s",
            e
        )

        raise


# ---------------------------------------------------------
# Save Metrics
# ---------------------------------------------------------

def save_metrics(
    metrics: dict,
    file_path: str
) -> None:
    """Save evaluation metrics to a JSON file."""

    try:

        with open(file_path, "w") as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        logging.info(
            "Metrics saved to %s",
            file_path
        )

    except Exception as e:

        logging.error(
            "Error occurred while saving the metrics: %s",
            e
        )

        raise


# ---------------------------------------------------------
# Save Model Information
# ---------------------------------------------------------

def save_model_info(
    model_info,
    file_path: str
) -> None:
    """
    Save MLflow logged model information.

    MLflow 3.x returns a ModelInfo object from
    mlflow.sklearn.log_model().
    """

    try:

        model_info_dict = {
            "model_id": model_info.model_id,
            "model_uri": model_info.model_uri,
            "run_id": model_info.run_id
        }

        with open(file_path, "w") as file:

            json.dump(
                model_info_dict,
                file,
                indent=4
            )

        logging.info(
            "Model information saved to %s",
            file_path
        )

    except Exception as e:

        logging.error(
            "Error occurred while saving model information: %s",
            e
        )

        raise


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # Set MLflow experiment
    mlflow.set_experiment(
        "my-dvc-pipeline"
    )

    # Start MLflow run
    with mlflow.start_run() as run:

        try:

            # -------------------------------------------------
            # Run ID
            # -------------------------------------------------

            print(
                "RUN ID:",
                run.info.run_id
            )


            # -------------------------------------------------
            # Load trained model
            # -------------------------------------------------

            clf = load_model(
                "./models/model.pkl"
            )


            # -------------------------------------------------
            # Load test data
            # -------------------------------------------------

            test_data = load_data(
                "./data/processed/test_bow.csv"
            )


            # -------------------------------------------------
            # Split X and y
            # -------------------------------------------------

            X_test = test_data.iloc[:, :-1].values

            y_test = test_data.iloc[:, -1].values


            # -------------------------------------------------
            # Evaluate model
            # -------------------------------------------------

            metrics = evaluate_model(
                clf,
                X_test,
                y_test
            )


            # -------------------------------------------------
            # Save metrics locally
            # -------------------------------------------------

            save_metrics(
                metrics,
                "reports/metrics.json"
            )


            # -------------------------------------------------
            # Log metrics to MLflow
            # -------------------------------------------------

            for metric_name, metric_value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value
                )


            # -------------------------------------------------
            # Log model parameters
            # -------------------------------------------------

            if hasattr(clf, "get_params"):

                params = clf.get_params()

                for param_name, param_value in params.items():

                    mlflow.log_param(
                        param_name,
                        param_value
                    )


            # -------------------------------------------------
            # Log model to MLflow
            # -------------------------------------------------

            model_info = mlflow.sklearn.log_model(
                clf,
                name="model"
            )

            print("MODEL LOGGED")

            print(
                "MODEL ID:",
                model_info.model_id
            )

            print(
                "MODEL URI:",
                model_info.model_uri
            )


            # -------------------------------------------------
            # Save MLflow model information
            # -------------------------------------------------

            save_model_info(
                model_info,
                "reports/experiment_info.json"
            )


            # -------------------------------------------------
            # Log metrics JSON as artifact
            # -------------------------------------------------

            mlflow.log_artifact(
                "reports/metrics.json"
            )


            # -------------------------------------------------
            # Final information
            # -------------------------------------------------

            logging.info(
                "Model evaluation completed successfully"
            )

            print(
                "Model evaluation completed successfully."
            )


        except Exception as e:

            logging.error(
                "Failed to complete the model evaluation process: %s",
                e
            )

            print(
                f"Error: {e}"
            )

            raise


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()