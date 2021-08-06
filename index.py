'''Abhi'''


from IPython.display import display
from PIL import Image
import datetime
import sys

import numpy as np
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


batch_size=32

def get_model():

    #intializing the convolution neural network
    classifier=Sequential()

    #step 1 - Convultion
    classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
    #Step 2 - MaxPooling
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    #Step 3 - Flattening
    classifier.add(Flatten())

    #Step 4 - Full Connection - connecting the convolution network to neural network
    classifier.add(Dense(128,activation='relu'))
    classifier.add(Dense(1,activation='sigmoid'))

    #Compiling the CNN
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return classifier




def data_augmenter():
    #Data augumentation - since neural net is not smart, even slight difference in images
    #shall be considered as seperate images thus to increase our training set size we shall
    # provide slight changes to images

    #part 2 - Fitting the CNN to the images


    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
    test_datagen= ImageDataGenerator(rescale=1./255)

    training_set=train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

    test_set=test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=batch_size,
                                            class_mode='binary')
    return (training_set, test_set)


def train(classifier, training_set, test_set, epochs):
    
    st = datetime.datetime.now()
    classifier.fit(
            training_set,
            steps_per_epoch=len(training_set),
            epochs=epochs,
            validation_data=test_set,
            validation_steps=len(test_set))

    end = datetime.datetime.now()
    print (f"Total Time: {(end - st).min} seconds")


def run(to_train):

    training_set, test_set = data_augmenter()

    if (to_train):
        classifier = get_model()
        train(classifier, training_set, test_set, 15)
        classifier.save('models/cat_dog_classifier')
    
    else:
        classifier = load_model('models/cat_dog_classifier')

    loss, accuracy = classifier.evaluate(test_set)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    test_image = image.load_img('dog.jpg',target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0]>=0.5:
        prediction='dog'
    else:
        prediction='cat'
    print(f"\n\nPrediction: {prediction}")




if __name__ == "__main__":

    to_train = len(sys.argv) == 2 and sys.argv[1] == "--train"
    run(to_train)
    
