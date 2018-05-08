import keras
from keras.layers.core import Lambda

def split_layer(layer, X):
    Y = []
    for x in X:
        Y.append(layer(x))
    return Y