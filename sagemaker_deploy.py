import mlflow
import mlflow.sagemaker
import click

@click.command()
@click.option("--app-name", type=click.STRING)
@click.option("--model-uri", type=click.STRING)
def run(app_name, model_uri):

    role = "arn:aws:iam::493356053890:role/service-role/AmazonSageMaker-ExecutionRole-20211006T173215"
    mlflow.sagemaker.deploy(app_name=app_name,
                    model_uri=model_uri,
                    execution_role_arn=role,
                    bucket="mlflow-data-asp",
                    image_url="493356053890.dkr.ecr.ap-south-1.amazonaws.com/mlflow-sagemaker-server:1.20.2",
                    region_name="ap-south-1",
                    instance_type="ml.m4.xlarge",
                    instance_count=1,
                    mode=mlflow.sagemaker.DEPLOYMENT_MODE_REPLACE,
                    timeout_seconds=1200)
if __name__ == "__main__":
    run()