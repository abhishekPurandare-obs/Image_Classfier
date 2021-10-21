#pip install -r requirements.txt -t .

import json
import numpy as np
from PIL import Image
import boto3
import os
import io

def predict(result):
    p = float(eval(result)[0][0])
    return "Cat" if p < 0.5 else "Dog"

def convert_image(image):
    test_image = Image.open(io.BytesIO(image)).resize((64, 64))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    print("SHAPE: ", test_image.shape)
    return test_image.tolist()

def send_sagemaker(sm_endpoint, data):

    region = os.environ['AWS_DEFAULT_REGION']
    sm = boto3.client('sagemaker', region_name=region)
    smrt = boto3.client('runtime.sagemaker', region_name=region)

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


def lambda_handler(event, context):


    try:
        image_bin = event['body']
        img_arr = convert_image(image_bin)
        data = {"inputs": img_arr}
        data = json.dumps(data)
        prediction = send_sagemaker("model-abhi1", data)

    except Exception as e:
        return {
            'statusCode': '502',
            'body': json.dumps(str(e))
        }       
    return {
        'statusCode': 200,
        'body': json.dumps(prediction)
    }
        
