#!/bin/bash
mlflow models serve -m $1 --port $2 --host $3 --no-conda