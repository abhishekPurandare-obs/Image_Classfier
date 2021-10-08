import mlflow
import tf_model
import click


@click.command()
@click.option("--model", type=click.STRING, default="1")
@click.option("--config", type=click.STRING, default="0")
def run(model, config):
    mlflow.set_tracking_uri("http://192.168.0.109:5000")
    print(f"\n\n\nURI:{mlflow.get_tracking_uri()}\n\n\n")
    tf_model.run(int(model.strip()), int(config.strip()))

if __name__ == "__main__":

    run()