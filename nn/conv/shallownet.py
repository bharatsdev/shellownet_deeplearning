from keras.models import Sequential
from keras.layers import Conv2D

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be "Channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using channel first we need to update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model
