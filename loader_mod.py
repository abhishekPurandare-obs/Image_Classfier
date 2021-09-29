import mlflow
import mlflow.tensorflow
class MyPredictModel(mlflow.pyfunc.PythonModel):


    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        
        # if isinstance(model_input, pd.DataFrame):
        #     raw_img_df = model_input.apply(base64.decodebytes, axis=1)
        #     images_df = raw_img_df.apply(self.decode_and_resize_image, axis=1)
        #     print(type(images_df["image"].values))
        #     print(type(images_df["image"].values[0]))
        prediction = "NULL"

        if isinstance(model_input, np.ndarray):

            result = self.model.predict(model_input)
            if result[0][0]>=0.5:
                    prediction='dog'
            else:
                    prediction='cat'
        return prediction

class _ModelWrapper:
    def __init__(self, model):

        self.model = model

    def predict(self, data):

        return type(data)

def _load_model(model_path):

    return mlflow.pyfunc.load_model(model_path)


def _load_pyfunc(model_path):

    keras_model = _load_model(model_path)
    return _ModelWrapper(keras_model)