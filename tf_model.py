import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import os
from argparse import Namespace
import datetime
import numpy as np

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization

import s3_transfer

if not tensorflow.test.is_gpu_available(cuda_only=True):
    os.environ["CUDA_VISIBLE_DEVICE"] = '-1'
#setting experiment name
#Keep the same name in all of the files to save runs under the same experiment.
# EXP_NAME = "Experiment-abhishek"
# EXP_NAME = os.environ["MLFLOW_EXP_NAME"]
# try:
#     EXP_ID = mlflow.create_experiment(EXP_NAME)
# except Exception as e:
#     print(e)
# mlflow.set_experiment(EXP_NAME)

client = MlflowClient()

#For this sample code, I am using each set of hyperparameters on two different models.
hyperparams1 = Namespace(
    batch_size = 32,
    conv1_feature_maps=64,
    max_pool1_size=(2, 2),
    conv2_feature_maps=32,
    learning_rate=2e-3,
    max_pool2_size=(2, 2),
    dense_layer1_size=128,
    dense_layer2_size=16,
    epochs=1
)

hyperparams2 = Namespace(
    batch_size = 32,
    conv1_feature_maps=32,
    max_pool1_size=(3, 3),
    conv2_feature_maps=16,
    max_pool2_size=(2, 2),
    dense_layer1_size=256,
    dense_layer2_size=32,
    learning_rate=0.01,
    epochs=1
)

CONFIG = [hyperparams1, hyperparams2]

#Keeping a custom callback for keras.
#Not used here though since I am only logging parameters after fitting the model.
#You can add mlflow params as per your requirements.
class CustomCallback(tensorflow.keras.callbacks.Callback):

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))


def get_model_1(config_idx=0):


    config = CONFIG[config_idx]
    #intializing the convolution neural network
    classifier=Sequential()

    #step 1 - Convultion
    classifier.add(Convolution2D(config.conv1_feature_maps, 3, 3,
                                input_shape=(64,64,3),
                                activation='relu'))
    #Step 2 - MaxPooling
    classifier.add(MaxPooling2D(pool_size=config.max_pool1_size))
    #Step 3 - Flattening
    classifier.add(Flatten())

    #Step 4 - Full Connection - connecting the convolution network to neural network
    classifier.add(Dense(config.dense_layer1_size,activation='relu'))
    classifier.add(Dense(1,activation='sigmoid'))

    #Compiling the CNN
    adam = Adam(learning_rate=config.learning_rate)
    classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

    return (classifier, "Model 1")


def get_model_2(config_idx=0):

    config = CONFIG[config_idx]
    #intializing the convolution neural network
    classifier=Sequential()

    classifier.add(Convolution2D(config.conv1_feature_maps, 5, 3,
                                activation='relu',
                                input_shape=(64, 64, 3)))
    classifier.add(BatchNormalization(axis=1, momentum=0.99, epsilon=5e-3))
    classifier.add(MaxPooling2D(pool_size=config.max_pool1_size))
    classifier.add(Convolution2D(config.conv2_feature_maps, 3, 3, activation='relu'))
    classifier.add(BatchNormalization(axis=1))
    classifier.add(MaxPooling2D(pool_size=config.max_pool2_size))

    classifier.add(Flatten())
    classifier.add(Dense(config.dense_layer1_size, activation='relu'))
    classifier.add(Dense(config.dense_layer2_size, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=config.learning_rate)
    classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return (classifier, "Model 2")

#performs image augmentation on the given datasets
def data_augmenter(config_idx=0):

    config = CONFIG[config_idx]

    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
    test_datagen= ImageDataGenerator(rescale=1./255)

    training_set=train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=config.batch_size,
                                                    class_mode='binary')

    test_set=test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=config.batch_size,
                                            class_mode='binary')
    return (training_set, test_set)

#This is a predict code for testing the model locally.
def predict(classifier, PATH="dog.jpg", get_signature=False):
    
    test_image = image.load_img(PATH,target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = classifier.predict(test_image)

    if result[0][0]>=0.5:
        prediction='dog'
    else:
        prediction='cat'
    print(f"\n\nPrediction: {prediction}")

    #signature is used by mlflow to identify input and output types of the model
    #This information will be stored along with the model.
    #When loading the model for running or serving. The I/O for the model will be enforced
    #based on the following saved types.
    if get_signature:
        return infer_signature(model_input=test_image, model_output=result)

def train(classifier, training_set, test_set, model_no=1, config_idx=0):
    

    config = CONFIG[config_idx]

    #saving the model name based on timestamp value
    model_path = "models/{:%d-%b-%y_%H-%M-%S}".format(datetime.datetime.now())
    print("Saving model at ", model_path)
    st = datetime.datetime.now()
    history = classifier.fit(
            training_set,
            steps_per_epoch=len(training_set),
            epochs=config.epochs,
            validation_data=test_set,
            validation_steps=len(test_set))
            # callbacks=[CustomCallback()])

    end = datetime.datetime.now()
    time_taken = (end - st).seconds

    signature = predict(classifier, get_signature=True)
    
    #History contains both losses and accuracies.
    #Returns a list of metrics so only saving the metrics of the last epoch
    for metric in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        mlflow.log_metric(metric, history.history[metric][-1])
    #Additional information to log
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("epochs", config.epochs)
    mlflow.log_param("time_taken", time_taken)
    mlflow.log_param("model_path", model_path)
    #autolog method will save most of the important parameters for the model
    mlflow.tensorflow.autolog()
    #Save the model under keras
    #Check documentation for different types of APIs depending on the library.
    mlflow.keras.save_model(classifier,
                            model_path,
                            # python_model=loader_mod.MyPredictModel(path),
                            signature=signature)

    # s3_transfer.save(model_path)
    # artifact_path = os.path.join("s3://"+os.environ['BUCKET']+"/artifacts", model_path)
    mlflow.log_artifacts(model_path)
    
def show_exp_details(exp_name):
    experiment = mlflow.get_experiment_by_name(exp_name)
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

def run(exp_name, model_no=1, config_idx=0):

    training_set, test_set = data_augmenter()

    if model_no == 1:
        classifier, model_name = get_model_1(config_idx)
    else:
        classifier, model_name = get_model_2(config_idx)

    print(model_name)
    print(classifier.summary())
    print("Active run: {}".format(mlflow.active_run()))

    run_name = f"Experiment: Pet Classifier MODEL {model_no} CONFIG {config_idx}"

    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    run = client.create_run(exp_id)
    run_id = run.info.run_uuid
    client.set_tag(run_id=run_id, key="mlflow.runName", value=run_name)

    #For each run setup in the following manner to let mlflow know when you
    #train the model.
    show_exp_details(exp_name)
    print(f"TRACKING URI:{mlflow.get_tracking_uri()}\n\n\n")
    with mlflow.start_run(experiment_id=exp_id, run_id=run_id, run_name=run_name) as run:
        mlflow.log_param("Experiment Name", run_name)
        train(classifier, training_set, test_set, model_no, config_idx)
    mlflow.end_run()

# if __name__ == "__main__":
    
    #Running both models on both configurations.
    # for model_no in range(1, 3):
        # for config_idx in range(len(CONFIG)):
    #       run(model_no, config_idx)
    # run()
    
