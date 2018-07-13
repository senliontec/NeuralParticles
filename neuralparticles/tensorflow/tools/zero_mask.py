import keras
import keras.backend as K
from keras.layers import Lambda

def zero_mask(inputs, mask_value):
    def tmp(v):
        mask = K.expand_dims(K.any(K.equal(v, mask_value), axis=-1), axis=-1)
        return 1 - K.cast(mask, K.floatx())
    return Lambda(tmp)(inputs)

def trunc_mask(trunc, size):
    def tmp(v):
        r = K.arange(size, dtype=K.floatx())+0.5
        return K.sigmoid((v*size-r)*10)
    return Lambda(tmp, name="truncation_mask")(trunc)