#!/bin/bash

bash exp_env.sh
mlflow run \
--docker-args env-file=project.env \
--experiment-name ${MLFLOW_EXP_NAME} \
--no-conda .