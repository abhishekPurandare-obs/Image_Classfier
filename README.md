# Image Classifier and MLFlow
This is a basic Pet classifier used for testing ML life cycle tracking and serving using MLflow.

## To serve the best model based on val_accuracy
```python serve_model.py```
by default model takes a request at ```<host>:<port>/invocations```

## To send a request
```python send_request.py --host http://0.0.0.0 --port 5000 --path dog.jpg```

## To build a docker image
```docker build -t mlflow_image .```

since the image has to be created beforehand to setup our environment. But we can specify a remote image repository as well.

## To run an experiment
```mlflow run --experiment-name Sample_Experiment . --no-conda --param-list model=1 config=0```

## Install mlflow
```pip install mlflow```

## To visualize your runs.
```mlflow ui```

By default, wherever you run your program, the tracking API writes data into a local ./mlruns directory. You can then run MLflow’s Tracking UI.
