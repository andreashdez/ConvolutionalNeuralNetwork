"""
Convolutional Neural Network using CIFAR10 training dataset.

file name: cnn_cifar10.py
author: Andreas Hernandez Hauser (anh45@aber.ac.uk)
description: A convolutional neural network build using the keras
             deep learning framework and trained on the CIFAR10 dataset.
"""
from itertools import chain

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


BATCH_SIZE = 128
NB_EPOCH = 15
DATA_AUGMENTATION = True
TRAIN = False
GREYSCALE = False


class Dataset:
    """Store details of the loaded dataset."""
    def __init__(self, nb_classes, img_rows, img_cols, img_channels,
                 validated_classes,
                 train_data, train_labels,
                 validate_data, validate_labels,
                 test_data, test_labels):
        self.nb_classes = nb_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.validated_classes = validated_classes
        self.train_data = train_data
        self.train_labels = train_labels
        self.validate_data = validate_data
        self.validate_labels = validate_labels
        self.test_data = test_data
        self.test_labels = test_labels


def rgb_to_greyscale(dataset_rgb):
    """
    Convert each image in the given dataset to greyscale.

    The dataset used in the model uses this specific shape:
        [channels, hight, width]
    it has to be changed as the rgb_to_hsv function needs this shape:
        [hight, width, channels]
    The new greyscale image is stored in a new array only using the last
    value of the hsv array.
    This new array has to be reshaped to meet the original criteria
    of the model.
    """
    dataset_grey = np.zeros((dataset_rgb.shape[0], 1,
                             dataset_rgb.shape[2], dataset_rgb.shape[3]))

    for i in range(len(dataset_rgb)):
        img_rgb = np.swapaxes(dataset_rgb[i], 0, 2)
        img_hsv = colors.rgb_to_hsv(img_rgb)
        img_grey = np.zeros((img_hsv.shape[0], img_hsv.shape[1]))
        for x in range(len(img_hsv)):
            for y in range(len(img_hsv[x])):
                img_grey[x][y] = img_hsv[x][y][2:]
        # plt.imshow(img_grey, cmap=cm.Greys_r)
        # plt.show()
        img_grey = img_grey.reshape(32, 32, 1)
        img_grey = np.swapaxes(img_grey, 2, 0)
        dataset_grey[i] = img_grey

    return dataset_grey


def load_data():
    """
    Load data using pre-configured dataset loader within keras.

    MNIST (http://yann.lecun.com/exdb/mnist/)
        - mnist.load_data()
    CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html)
        - cifar10.load_data()
    CIFAR100 (https://www.cs.toronto.edu/~kriz/cifar.html)
        - cifar100.load_data()
    """
    # The data, shuffled and split between train and test sets
    (X_data, train_labels), (y_data, test_labels) = cifar10.load_data()

    X_data = X_data.astype('float32')/255
    y_data = y_data.astype('float32')/255

    if GREYSCALE:
        train_data = rgb_to_greyscale(X_data)
        test_data = rgb_to_greyscale(y_data)
    else:
        train_data = X_data
        test_data = y_data

    # Number of classes in the used dataset
    nb_classes = 10
    # input image dimensions
    img_rows, img_cols = 32, 32
    # RGB images have 3 channels and greyscale images have 1
    img_channels = test_data.shape[1]

    # Static creation of training and validating datasets
    validate_data_id = np.random.randint(train_data.shape[0], size=10000)
    validate_data = train_data[validate_data_id, :]
    validate_labels = train_labels[validate_data_id, :]
    train_data_id = np.random.randint(train_data.shape[0], size=40000)
    train_data = train_data[train_data_id, :]
    train_labels = train_labels[train_data_id, :]

    # create flat array of test labels for evaluation of the model
    validated_classes = list(chain.from_iterable(test_labels))

    # convert class vectors to binary class matrices
    train_labels = np_utils.to_categorical(train_labels, nb_classes)
    validate_labels = np_utils.to_categorical(validate_labels, nb_classes)
    test_labels = np_utils.to_categorical(test_labels, nb_classes)

    dataset = Dataset(nb_classes, img_rows, img_cols, img_channels,
                      validated_classes,
                      train_data, train_labels,
                      validate_data, validate_labels,
                      test_data, test_labels)
    return dataset


def create_cnn(dataset):
    """Create convolutional neural network model."""
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            input_shape=(dataset.img_channels,
                                         dataset.img_rows, dataset.img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dataset.nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6,
                                momentum=0.9, nesterov=True))
    return model


def train_model(model, dataset):
    """
    Train convolutional neural network model.

    Provides the option of using data augmentation to minimize over-fitting.
    Options used currently are:
        rotation_range - rotates the image.
        width_shift_range - shifts the position of the image horizontally.
        height_shift_range - shifts the position of the image vertically.
        horizontal_flip - flips the image horizontally.
    """
    print("\n- TRAINING MODEL -----------------------------------------------")
    if not DATA_AUGMENTATION:
        print('Not using data augmentation.')
        model.fit(dataset.train_data, dataset.train_labels,
                  batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, shuffle=True,
                  verbose=1, show_accuracy=True,
                  validation_data=(dataset.validate_data,
                                   dataset.validate_labels))
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            # Rotate image between 0 and 10 degrees randomly
            rotation_range=0.1,
            # Shift image by 1px horizontally randomly
            width_shift_range=0.1,
            # Shift image by 1px vertically randomly
            height_shift_range=0.1,
            # Flip the image horizontally randomly
            horizontal_flip=True,
            vertical_flip=False)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(dataset.train_data)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(dataset.train_data,
                                         dataset.train_labels,
                                         shuffle=True, batch_size=BATCH_SIZE),
                            samples_per_epoch=dataset.train_data.shape[0],
                            nb_epoch=NB_EPOCH, verbose=1, show_accuracy=True,
                            validation_data=(dataset.validate_data,
                                             dataset.validate_labels),
                            nb_worker=1)
    return model


def evaluate_model(model, dataset):
    """
    Evaluate convolutional neural network model.

    Evaluate on the test data which has not been used in the training process.
    Creates a confusion matrix and a classification report
    to aid evaluation.
    """
    print("\n- EVALUATING MODEL ---------------------------------------------")
    score = model.evaluate(dataset.test_data, dataset.test_labels,
                           batch_size=BATCH_SIZE,
                           verbose=1, show_accuracy=True)
    print(score)

    predicted_classes = model.predict_classes(dataset.test_data,
                                              verbose=0).tolist()

    print("\nConfusion matrix:")
    conf_matrix = confusion_matrix(dataset.validated_classes,
                                   predicted_classes)
    print(conf_matrix)
    plt.figure()
    plot_confusion_matrix(conf_matrix)
    plt.show()

    print("\nClassification report:")
    print(classification_report(dataset.validated_classes, predicted_classes))


def plot_confusion_matrix(conf_matrix, title='Confusion matrix'):
    plt.imshow(conf_matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.ylabel('Real values')
    plt.xlabel('Predicted values')


if __name__ == "__main__":
    dataset = load_data()
    model = create_cnn(dataset)
    if TRAIN:
        model = train_model(model, dataset)
    evaluate_model(model, dataset)
