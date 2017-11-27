import matplotlib
matplotlib.use('Agg')
from helpers import *
from dataset import Dataset
import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from subpixel import *

import numpy as np

paramUsed = []

src_patches_path = getParam("src", "", paramUsed)
ref_patches_path = getParam("ref", "", paramUsed)

data_start = int(getParam("data_start", 0, paramUsed))
data_end = int(getParam("data_end", 0, paramUsed))

time_start = int(getParam("time_start", 0, paramUsed))
time_end = int(getParam("time_end", 0, paramUsed))

var = int(getParam("var", 0, paramUsed))

features = getParam("features", "", paramUsed).split(",")


val_split = float(getParam("val_split", 0., paramUsed))
epochs = int(getParam("epochs", 0, paramUsed))
batch_size = int(getParam("batch", 0, paramUsed))

log_intervall = int(getParam("log_intervall", 10, paramUsed))
checkpoint_intervall = int(getParam("checkpoint_intervall", 10, paramUsed))

start_checkpoint = int(getParam("start_checkpoint", 0, paramUsed))

model_src = getParam("model", "", paramUsed)
fig_path = getParam("fig", "", paramUsed)

checkUnusedParam(paramUsed)

print("Load Training Data")
train_data = Dataset(src_patches_path, 
                     data_start, data_end, time_start, time_end, 
                     features, var, ref_patches_path, ['sdf'])

print("Source Data Shape: " + str(train_data.data.shape))
print("Reference Data Shape: " + str(train_data.ref_data.shape))

class NthLogger(keras.callbacks.Callback):
    def __init__(self,li=10,cpi=100,model_path="model"):
        self.act = 0
        self.li = li
        self.cpi = cpi
        self.model_path = model_path

    def on_epoch_end(self,batch,logs={}):
        self.act += 1
        if self.act % self.li == 0 or self.act == 1:
            print('%d/%d - loss: %f val_loss: %f' % (self.act, self.params['epochs'], logs['loss'], logs['val_loss']))
        if self.act % self.cpi == 0:
            path = "%s_%04d.h5" % (self.model_path, self.act//self.cpi)
            model.save(path)
            print('Saved Checkpoint: %s' % path)

if start_checkpoint > 0:
    m_p = "%s_%04d.h5" % (model_src, start_checkpoint)
else:
    m_p = "%s.h5" % model_src

print("Loading Model: %s" % m_p)
model = load_model(m_p, custom_objects={'Subpixel': Subpixel})

print("Start Training")
history = model.fit(x=np.split(train_data.data,[1],axis=3),y=train_data.ref_data, validation_split=val_split, 
                    epochs=epochs - start_checkpoint*checkpoint_intervall, batch_size=batch_size, 
                    verbose=0, callbacks=[NthLogger(log_intervall, checkpoint_intervall, model_src)])

m_p = "%s_trained.h5" % model_src
model.save(m_p)
print("Saved Model: %s" % m_p)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.savefig(fig_path+".png")
plt.savefig(fig_path+".pdf")
