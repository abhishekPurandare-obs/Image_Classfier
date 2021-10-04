# Image_Classfier and MLFlow
This is a basic image classifer to try out Machine learning life cycle using MLFlow.

## To serve the best model
```python serve_model.py```
by default model takes request at ```<host>:<port>/invocations```

## To send a request
```python send_request.py --host http://0.0.0.0 --port 5000 --path dog.jpg```

## To build a docker image
```docker build -t mlflow_image .```

## To run an experiment
```mlflow run --experiment-name "Sample Experiment" ---docker-args env-file=project.env . --no-conda --param-list model=1 config=0```