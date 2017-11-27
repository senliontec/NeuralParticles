import matplotlib
matplotlib.use('Agg')
from helpers import *
from dataset import Dataset
import matplotlib.pyplot as plt

import keras
from keras.models import load_model, Model
from keras.layers import Input
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

mse_fac = float(getParam("mse", 1.0, paramUsed))
adv_fac = float(getParam("adv", 0.1, paramUsed))

lr = float(getParam("lr", 0.001, paramUsed))

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

if start_checkpoint > 0:
    gen_p = "%s_%04d.h5" % (model_src, start_checkpoint)
    adv_p = "%s_dis_%04d.h5" % (model_src, start_checkpoint)
else:
    gen_p = "%s.h5" % model_src
    adv_p = "%s_dis.h5" % (model_src)

print("Loading Model: %s" % gen_p)
generator = load_model(gen_p, custom_objects={'Subpixel': Subpixel})
discriminator = load_model(adv_p)

z = Input(shape=(train_data.data.shape[1],train_data.data.shape[2],1), name='main')
z_aux = Input(shape=(train_data.data.shape[1],train_data.data.shape[2],train_data.data.shape[3]-1), name='aux')
img = generator([z,z_aux])

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
combined = Model([z,z_aux], [img,valid])
combined.compile(loss=['mse','binary_crossentropy'], optimizer=keras.optimizers.adam(lr=lr),
                loss_weights=[mse_fac, adv_fac])

print("Start Training")

half_batch = batch_size//2

train_cnt = int(len(train_data.data)*(1-val_split))//batch_size*batch_size
print('train count: %d' % train_cnt)
eval_cnt = int(len(train_data.data)*val_split)//batch_size*batch_size
print('eval count: %d' % eval_cnt)

cnt_inv = batch_size/train_cnt

history = {'d_loss':[],'d_acc':[],'g_mse':[],'g_loss':[],
           'd_val_loss':[],'d_val_acc':[],'g_val_mse':[],'g_val_loss':[]}
idx0 = np.arange(train_cnt+eval_cnt)
idx1 = np.arange(train_cnt+eval_cnt)

np.random.shuffle(idx0)
np.random.shuffle(idx1)
    
idx0, val_idx0 = np.split(idx0,[train_cnt])
idx1, val_idx1 = np.split(idx1,[train_cnt])

for ep in range(epochs):    
    # train
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    g_loss = [0.,0.,0.]
    d_loss = [0.,0.]
    
    for i in range(0,train_cnt,batch_size):
        x = train_data.ref_data[idx0[i:i+half_batch]]
        
        y = train_data.data[idx0[i+half_batch:i+batch_size]]
        y = np.split(y,[1],axis=3)
        y = generator.predict(y)

        d_loss_real = discriminator.train_on_batch(x, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(y, np.zeros((half_batch, 1)))
        d_loss = np.add(d_loss, cnt_inv * 0.5 * np.add(d_loss_real, d_loss_fake) )
        
        x = train_data.data[idx1[i:i+batch_size]]
        y = train_data.ref_data[idx1[i:i+batch_size]]
        x = np.split(x,[1],axis=3)
        g_loss = np.add(g_loss, cnt_inv * np.array(combined.train_on_batch(x, [y,np.ones((batch_size, 1))])))
    
    # eval
    np.random.shuffle(val_idx0)
    np.random.shuffle(val_idx1)
    g_val_loss = [0.,0.,0.]
    d_val_loss = [0.,0.]
    
    x = train_data.ref_data[val_idx0]
    
    y = train_data.data[val_idx0]
    y = np.split(y,[1],axis=3)
    y = generator.predict(y)
    
    d_loss_real = discriminator.evaluate(x, np.ones((eval_cnt, 1)), batch_size=half_batch, verbose=0)
    d_loss_fake = discriminator.evaluate(y, np.zeros((eval_cnt, 1)), batch_size=half_batch, verbose=0)
    d_val_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    x = train_data.data[val_idx1]
    y = train_data.ref_data[val_idx1]
    x = np.split(x,[1],axis=3)
    
    g_val_loss = combined.evaluate(x, [y,np.ones((eval_cnt, 1))], batch_size=batch_size, verbose=0)
    
    history['d_loss'].append(d_loss[0])
    history['d_acc'].append(d_loss[1])
    history['d_val_loss'].append(d_val_loss[0])
    history['d_val_acc'].append(d_val_loss[1])
    history['g_mse'].append(g_loss[0])
    history['g_loss'].append(g_loss[1])
    history['g_val_mse'].append(g_val_loss[0])
    history['g_val_loss'].append(g_val_loss[1])
            
    if ep+1 % log_intervall == 0 or ep == 1:
        print ("epoch %i" % (ep+1))
        print ("\ttrain: [D loss: %f, acc.: %.2f%%] [G mse: %f, loss: %f]" % (d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
        print ("\teval.: [D loss: %f, acc.: %.2f%%] [G mse: %f, loss: %f]" % (d_val_loss[0], 100*d_val_loss[1], g_val_loss[0], g_val_loss[1]))
    
    if ep+1 % checkpoint_intervall == 0:
        path = "%s_%04d.h5" % (model_src, (ep+1)//checkpoint_intervall)
        generator.save(path)
        print('Saved Generator Checkpoint: %s' % path)
        path = "%s_%04d_dis.h5" % (model_src, (ep+1)//checkpoint_intervall)
        discriminator.save(path)
        print('Saved Generator Checkpoint: %s' % path)

gen_p = "%s_trained.h5" % model_src
generator.save(gen_p)
print("Saved Model: %s" % gen_p)

plt.plot(history['g_mse'])
plt.plot(history['g_loss'])
plt.plot(history['g_val_mse'])
plt.plot(history['g_val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['g_mse','g_loss','g_val_mse','g_val_loss'], loc='upper left')

plt.savefig(fig_path+".png")
plt.savefig(fig_path+".pdf")

plt.plot(history['d_loss'])
plt.plot(history['d_val_loss'])
plt.title('discriminator loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['d_loss','d_val_loss'], loc='upper left')

plt.savefig(fig_path+"_dis.png")
plt.savefig(fig_path+"_dis.pdf")
