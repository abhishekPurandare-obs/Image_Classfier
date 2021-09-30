import os
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
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

mlflow.set_experiment(experiment_name="Sample experiment")

hyperparams1 = Namespace(
    batch_size = 32,
    conv1_feature_maps=64,
    max_pool1_size=(2, 2),
    conv2_feature_maps=32,
    learning_rate=2e-3,
    max_pool2_size=(2, 2),
    dense_layer1_size=128,
    dense_layer2_size=16,
    epochs=10
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
    epochs=10
)

CONFIG = [hyperparams1, hyperparams2]



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

    if get_signature:
        return infer_signature(model_input=test_image, model_output=result)

def train(classifier, training_set, test_set, model_no=1, config_idx=0):
    

    config = CONFIG[config_idx]
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
    mlflow.log_params(history.history)
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("epochs", config.epochs)
    mlflow.log_param("time_taken", time_taken)
    mlflow.log_param("model_path", model_path)
    mlflow.tensorflow.autolog()
    mlflow.keras.save_model(classifier,
                            model_path,
                            # python_model=loader_mod.MyPredictModel(path),
                            signature=signature)


def run_model(model_no=1, config_idx=0):

    training_set, test_set = data_augmenter()

    if model_no == 1:
        classifier, model_name = get_model_1(config_idx)
    else:
        classifier, model_name = get_model_2(config_idx)

    print(model_name)
    print(classifier.summary())

    exp_name = f"Experiment: Pet Classifier MODEL {model_no} CONFIG {config_idx}"
    with mlflow.start_run(run_name=exp_name) as run:

        run_id = run.info.run_uuid
        exp_id = run.info.experiment_id
        print(f"*****Running Run {run_id} Experiment {exp_id}*****")
        mlflow.log_param("Experiment Name", exp_name)
        train(classifier, training_set, test_set, model_no, config_idx)

if __name__ == "__main__":

    for model_no in range(1, 3):
        for config_idx in range(len(CONFIG)):
            run_model(model_no, config_idx)
    
