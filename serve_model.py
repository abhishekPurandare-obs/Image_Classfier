import mlflow
import mlflow.tensorflow
from mlflow.tracking.client import MlflowClient
from tensorflow.keras.preprocessing import image
import os
import numpy as np

client = MlflowClient()



def print_exp_info(exp):
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))

def get_runs(exp):

    return [d for d in os.listdir(exp.artifact_location[7:])
    if os.path.isdir(os.path.join(exp.artifact_location[7:], d))]

#get best model from all the runs
def get_best_model_id(exp, metric_name="val_accuracy"):

    run_ids = get_runs(exp)
    best_run = ""
    best_metric = float('inf') if "loss" in metric_name else float('-inf')
    metrics = {}
    for r_id in run_ids:
        try:
            metrics_dict = client.get_run(r_id).data.metrics
            val_accuracy = metrics_dict['val_accuracy']
            val_loss = metrics_dict['val_loss']
            model_path = client.get_run(r_id).data.params["model_path"]
            metrics[r_id] = {'val_accuracy': float(val_accuracy),
                            'val_loss': float(val_loss),
                            'model_path': model_path}
        except Exception as e:
            print(e)
            print(f"Skipping {r_id}")

    for r_id, metric in metrics.items():
        m = metric[metric_name]
        if ("loss" in metric_name and best_metric > m) or best_metric < m:
            best_run  = r_id
            best_metric = m

    print(f"Best Run:{best_run}\n{metric_name}: {best_metric}")

    return (best_run, metrics[best_run])



#serve the model locally
def serve_model(model_path, port=5000, host="0.0.0.0"):

    command = f"sh serve.sh {model_path} {port} {host}"
    os.system(command)

if __name__ == "__main__":
    
    # exp_name = os.environ["MLFLOW_EXP_NAME"]
    exp_name = "Sample_Experiment"
    exp = client.get_experiment_by_name(exp_name)
    print_exp_info(exp)
    best_run_id, best_run_metrics = get_best_model_id(exp)
    model_path = best_run_metrics["model_path"]
    print("MODEL: ", model_path)

    model = mlflow.pyfunc.load_model(model_path)
    image_name = "dog.jpg"
    test_image = image.load_img(image_name, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    print(model.predict(test_image))
    serve_model(model_path)