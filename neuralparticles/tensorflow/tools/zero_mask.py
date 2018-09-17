import keras
import keras.backend as K
from keras.layers import Lambda

def zero_mask(inputs, mask_value, **kwargs):
    def tmp(v):
        mask = K.expand_dims(K.any(K.equal(v, mask_value), axis=-1), axis=-1)
        return 1 - K.cast(mask, K.floatx())
    return Lambda(tmp, **kwargs)(inputs)

def trunc_mask(trunc, size, **kwargs):
    def tmp(v):
        r = K.arange(size, dtype=K.floatx())
        mask = K.less(r, trunc)
        return K.cast(mask, K.floatx())
    return Lambda(tmp, **kwargs)(trunc)

def soft_trunc_mask(trunc, size, steepness=10,**kwargs):
    def tmp(v):
        r = K.arange(size, dtype=K.floatx())+0.5
        return K.sigmoid((v*size-r)*steepness)
    return Lambda(tmp, **kwargs)(trunc)