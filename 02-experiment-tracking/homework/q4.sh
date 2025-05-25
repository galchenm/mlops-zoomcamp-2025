#!/bin/bash

# Define paths
BACKEND_URI="sqlite:///mlflow_q4.db"
ARTIFACT_ROOT="./artifacts"
HOST="0.0.0.0"
PORT="5000"

# Create artifacts directory if it doesn't exist
mkdir -p $ARTIFACT_ROOT

# Launch the MLflow tracking server
mlflow server \
  --backend-store-uri $BACKEND_URI \
  --default-artifact-root $ARTIFACT_ROOT \
  --host $HOST \
  --port $PORT
