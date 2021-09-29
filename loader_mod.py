import mlflow
import numpy as np
import pickle
import os

class MyPredictModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_path):
        self.model_path = os.path.join(model_path, "python_model.pkl")
        self.model = None

    def load_model(self):
        self.model = pickle.load(open(self.model_path, "rb"))

    def predict(self, context, model_input):

        prediction = type(model_input)
        self.load_model()

        # if isinstance(model_input, np.ndarray):
            
            # result = self.model.predict(model_input)
            # if result[0][0]>=0.5:
            #         prediction='dog'
            # else:
            #         prediction='cat'

        d = str(dir(self.model))
        return str(type(self.model)) + "\n" + d

class _ModelWrapper:
    def __init__(self, model):

        self.model = model

    def predict(self, data):

        return type(data)

def _load_model(model_path):

    return mlflow.pyfunc.load_model(model_path)


# def _load_pyfunc(model_path):

#     keras_model = _load_model(model_path)
#     return _ModelWrapper(keras_model)


if __name__ == "__main__":
    pass