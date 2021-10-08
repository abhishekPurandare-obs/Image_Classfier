mlflow run \
    --docker-args env-file=project.env \
    --docker-args network="host" \
    --experiment-name Sample_Experiment \
    --no-conda . 