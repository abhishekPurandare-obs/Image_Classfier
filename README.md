# Image Classifier and MLFlow
This is a basic Pet classifier used for testing ML life cycle tracking and serving using MLflow.

## Install mlflow
```pip install mlflow```

## To build a docker image
```docker build -t mlflow_image .```

since the image has to be created beforehand to setup our environment. But we can specify a remote image repository as well.

## To run an experiment
```mlflow run --experiment-name Sample_Experiment . --no-conda --param-list model=1 config=0```

You can also make a run by specifying git repo
```mlflow run https://github.com/abhishekPurandare-obs/mlflow_test.git --no-conda -P  model=1 -P config=0```

```mlflow run``` looks for ```MLProject``` to configure environment for your current run.



## To serve the best model based on val_accuracy
```python serve_model.py```
by default model takes a request at ```<host>:<port>/invocations```

## To send a request
```python send_request.py --host http://0.0.0.0 --port 5000 --path dog.jpg```

## To visualize your runs
```mlflow ui```

By default, wherever you run your program, the tracking API writes data into a local ./mlruns directory. You can then run MLflow’s Tracking UI.

## To serve the model as a docker image

We can also serve the model using a container by executing the following command

```mlflow models build-docker -m "path/to/model" -n "image-name"```

```docker run -t "image-name" -p 5000:8080```

I haven't tried this yet because the endpoint of the API created by mlflow takes a specific input [Here](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlmodel-configuration)

In this example, I am first preprocessing the image in [send_request.py](send_request.py) to make it compatible with the expected input by the endpoint. But there is a way to create a custom predict function for specific use-case too. So that way, we can send any type of image. That will be passed to your custom predict routine and then make a prediction.
