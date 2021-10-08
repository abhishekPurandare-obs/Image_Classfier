import os
import boto3

AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

BUCKET = os.environ['BUCKET']


s3_resource = boto3.resource(
    's3', 
    region_name = AWS_DEFAULT_REGION,
    aws_access_key_id = AWS_ACCESS_KEY_ID, 
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY)


def save_to_s3(path):

    user_id = "Abhishek"
    fname = os.path.split(path)[-1]
    KEY = os.path.join(f"{user_id}/models",fname)
    s3_resource.Bucket(BUCKET).put_object(
        Key = KEY,
        Body = open(path, 'rb'))