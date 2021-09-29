import mlflow
import mlflow.tensorflow
from mlflow.tracking.client import MlflowClient
import tf_model
import os

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
            params = client.get_run(r_id).data.params
            val_accuracy = eval(params['val_accuracy'])[-1]
            val_loss = eval(params['val_loss'])[-1]
            model_path = params["model_path"]
            # model_path = os.path.join(params['model_path'], "python_model.pkl")
            metrics[r_id] = {'val_accuracy': float(val_accuracy),
                            'val_loss': float(val_loss),
                            'model_path': model_path}
        except:
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

    exp = client.get_experiment_by_name("Sample experiment")
    print_exp_info(exp)
    best_run_id, best_run_metrics = get_best_model_id(exp)
    # model = _load_pyfunc(best_run_metrics["model_path"])

    serve_model(best_run_metrics["model_path"])