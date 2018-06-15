import keras
import keras.backend as K

def zero_mask(inputs, mask_value):
    mask = K.expand_dims(K.any(K.equal(inputs, mask_value), axis=-1), axis=-1)
    return 1 - K.cast(mask, K.floatx())

def trunc_mask(trunc, size):
    r = K.arange(size, dtype=K.floatx())+0.5
    return K.sigmoid((trunc*size-r)*10)