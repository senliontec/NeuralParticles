import keras
import keras.backend as K

def holographic(y_true, y_pred, n):
    yt = K.expand_dims(y_true, 1)
    yp = K.expand_dims(y_pred, 2)

    yt = K.repeat_elements(yt, n, axis=1)
    yp = K.repeat_elements(yp, n, axis=2)
    dist = K.sum(((yt-yp)/2)**2, axis=-1)
    return K.sum(K.prod(dist/K.max(dist,axis=(1,2)), axis=2), axis=1)