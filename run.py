import mlflow
import tf_model
import click

@click.command()
@click.option("--model", type=click.STRING, default="1")
@click.option("--config", type=click.STRING, default="0")
def run(model, config):
    tf_model.run(int(model.strip()), int(config.strip()))

if __name__ == "__main__":

    run()