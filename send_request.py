import requests
import json
import boto3
import click
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def predict(result):
    p = float(eval(result)[0][0])
    return "Cat" if p < 0.5 else "Dog"

def convert_image(image_name):
    test_image = image.load_img(image_name, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    print("SHAPE: ", test_image.shape)
    return test_image.tolist()


def send_local(host, port, data):
    response = requests.post(
        url=f"{host}:{port}/invocations",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        raise Exception(
            "Status Code {status_code}. {text}".format(
                status_code=response.status_code, text=response.text
            )
        )

    return response.text


def send_sagemaker(sm_endpoint, data):

    region = os.environ['AWS_DEFAULT_REGION']
    sm = boto3.client('sagemaker', region_name=region)
    # smrt = boto3.client('runtime.sagemaker', region_name=region)
    smrt = boto3.client("sagemaker-runtime", region_name=region)

    # Check endpoint status
    endpoint = sm.describe_endpoint(EndpointName=sm_endpoint)
    print("Endpoint status: ", endpoint["EndpointStatus"])
    
    prediction = smrt.invoke_endpoint(
        EndpointName=sm_endpoint,
        Body=data,
        ContentType='application/json'
    )

    prediction = prediction['Body'].read().decode("ascii")
    print(type(prediction))

    return prediction


@click.command()
@click.option("--port", type=click.INT, default=5000)
@click.option("--host", type=click.STRING, default="http://0.0.0.0")
@click.option("--path", type=click.STRING, default="dog.jpg")
@click.option("--sm-endpoint", type=click.STRING)
def show_prediction(path, port, host, sm_endpoint):

    test_img = convert_image(path)
    data = {"inputs": test_img}
    data = json.dumps(data)

    # prediction = predict(send_local(host, port, data))

    prediction = send_sagemaker(sm_endpoint, data)
    print("Prediction: ", predict(prediction))
    



if __name__ == "__main__":

    show_prediction()