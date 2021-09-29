import requests
import json
import os
import click
from tensorflow.keras.preprocessing import image
import numpy as np

@click.command()
@click.option("--port", type=click.INT, default=5000)
@click.option("--host", type=click.STRING, default="http://0.0.0.0")
@click.option("--path", type=click.STRING, default="dog.jpg")
def send(path, port, host):


    # if os.path.isdir(path):
    #     filenames = [
    #         os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))
    #     ]
    # else:
    #     filenames = [path]

    def convert_image(image_name):
        test_image = image.load_img(image_name, target_size=(64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        print("SHAPE: ", test_image.shape)
        return test_image.tolist()

    test_img = convert_image(path)
    data = {"inputs": test_img}
    data = json.dumps(data)
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

    return response


if __name__ == "__main__":

    res = send()
    pred = "Cat" if float(res.text) < 0.5 else "Dog"
    print("Output: ", pred)