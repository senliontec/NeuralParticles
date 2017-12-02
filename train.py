import sys, os
sys.path.append("manta/scenes/tools")

import json
from helpers import *

from dataset import Dataset

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, ZeroPadding2D, Dense
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from subpixel import *
from keras import regularizers
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0

log_intervall = int(getParam("log_intervall", 10, paramUsed))
checkpoint_intervall = int(getParam("checkpoint_intervall", 10, paramUsed))

start_checkpoint = int(getParam("start_checkpoint", 0, paramUsed))

src_path = data_path + "patches/source/"
ref_path = data_path + "patches/reference/"

model_path = data_path + "models/"
if not os.path.exists(model_path):
	os.makedirs(model_path)

checkpoint_path = model_path + "checkpoints/"
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)

with open(config_path, 'r') as f:
    config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['data'], 'r') as f:
    data_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['preprocess'], 'r') as f:
    pre_config = json.loads(f.read())

with open(os.path.dirname(config_path) + '/' + config['train'], 'r') as f:
    train_config = json.loads(f.read())

if verbose:
    print("Config Loaded:")
    print(config)
    print(data_config)
    print(pre_config)
    print(train_config)

features = train_config['features']
feature_cnt = len(features)
if 'vel' in features:
    feature_cnt += 2
print("feature_count: %d" % feature_cnt)

model_path = '%s%s_%s' % (model_path, data_config['prefix'], config['id'])
checkpoint_path = '%s%s_%s' % (checkpoint_path, data_config['prefix'], config['id'])
print(model_path)
fig_path = '%s_loss' % model_path

src_path = "%s%s_%s-%s" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
ref_path = "%s%s_%s-%s" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_%03d"
print(src_path)
print(ref_path)

if start_checkpoint == 0:
    print("Generate Network")
    if train_config['explicit']:  
        inputs = Input((pre_config['par_cnt'],3), name="main")
        auxiliary_input = Input(shape=(pre_config['patch_size'], pre_config['patch_size'], feature_cnt-1), name="auxiliary_input")  
        
        base = Flatten()(par_in)
        base = Dense(100, activation='tanh')(base)
        
        x = Conv2D(filters=16, kernel_size=3, 
                strides=1, activation='tanh', padding='same', name="conv2D_0")(auxiliary_input)
        x = BatchNormalization(name="normalize_0")(x)
        x = Conv2D(filters=32, kernel_size=3,
                strides=1, activation='tanh', padding='same', name="conv2D_1")(x)    
        x = BatchNormalization(name="normalize_1")(x)
        x = Conv2DTranspose(filters=16, kernel_size=3, 
                            strides=1, activation='tanh', padding='same', name="deconv2D_0")(x)
        x = BatchNormalization(name="normalize_2")(x)
        x = Conv2DTranspose(filters=4, kernel_size=3, 
                            strides=1, activation='tanh', padding='same', name="deconv2D_1")(x)
        x = BatchNormalization(name="normalize_3")(x)
        
        x = Reshape((pre_config['patch_size']*pre_config['patch_size']*4,))(x)
        
        base = concatenate([base, x], name="concatenate")
        
        base = Dense(100, activation='tanh')(base)
        base = Dense(pre_config['par_cnt']*3, activation='tanh')(base)
        out = Reshape((pre_config['par_cnt'],3))(base)
        
        model = Model(inputs=[par_in, auxiliary_input], outputs=out)
        model.compile( loss='mse', optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]))
        
        model.save(model_path + '.h5')
        
        if verbose: 
            model.summary()
    else:   
        inputs = Input((pre_config['patch_size'], pre_config['patch_size'], 1), name="main_input")
        auxiliary_input = Input(shape=(pre_config['patch_size'], pre_config['patch_size'], feature_cnt-1), name="auxiliary_input")

        base = Reshape((pre_config['patch_size']*pre_config['patch_size'],), name="reshape_flat")(inputs)
        base = RepeatVector(9, name="repeate")(base)
        base = Permute((2, 1), name="permute")(base)
        base = Reshape((pre_config['patch_size'], pre_config['patch_size'],9), name="reshape_back")(base)
        
        x = concatenate([inputs, auxiliary_input], name="concatenate")
        x = Reshape((pre_config['patch_size']*pre_config['patch_size']*feature_cnt,), name="reshape_flat_res")(x)
        x = RepeatVector(9, name="repeate_res")(x)
        x = Permute((2, 1), name="permute_res")(x)
        x = Reshape((pre_config['patch_size'], pre_config['patch_size'],9*feature_cnt), name="reshape_back_res")(x)
        
        #x = concatenate([base, auxiliary_input], name="concatenate")
        
        x = Conv2D(filters=16*feature_cnt, kernel_size=3, 
                strides=1, activation='tanh', padding='same', name="conv2D_0")(x)
        x = BatchNormalization(name="normalize_0")(x)
        x = Conv2D(filters=32*feature_cnt, kernel_size=3,
                strides=1, activation='tanh', padding='same', name="conv2D_1")(x)    
        x = BatchNormalization(name="normalize_1")(x)
        x = Conv2DTranspose(filters=16*feature_cnt, kernel_size=3, 
                            strides=1, activation='tanh', padding='same', name="deconv2D_0")(x)
        x = BatchNormalization(name="normalize_2")(x)
        x = Conv2DTranspose(filters=9, kernel_size=3, 
                            strides=1, activation='tanh', padding='same', name="deconv2D_1")(x)
        x = BatchNormalization(name="normalize_3")(x)
        
        x = add([base,x], name="add")
        x = Activation('tanh', name="activation")(x)
        predictions = Subpixel(filters=1, kernel_size=3, r=3,activation='tanh', padding='same', name="subpixel_conv")(x)
        
        model = Model(inputs=[inputs,auxiliary_input], outputs=predictions, name="generator")
        model.compile( loss='mse', optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]))
        
        model.save(model_path + '.h5')
        
        if verbose: 
            model.summary()

        if train_config["adv_fac"] > 0.:
            generator = model
            discriminator = Sequential(name="discriminator")

            img_shape = (high_patch_size, high_patch_size, 1)

            discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
            discriminator.add(LeakyReLU(alpha=0.2))
            discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
            discriminator.add(LeakyReLU(alpha=0.2))
            discriminator.add(BatchNormalization(momentum=0.8))
            discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
            discriminator.add(LeakyReLU(alpha=0.2))
            discriminator.add(BatchNormalization(momentum=0.8))

            discriminator.add(Flatten())
            discriminator.add(Dense(1, activation='sigmoid'))

            discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]), metrics=['accuracy'])

            discriminator.save(model_path + '_dis.h5')

            if verbose: 
                discriminator.summary()
else:
    if not train_config["adv_fac"] > 0.:
        model = load_model("%s_%04d.h5" % (checkpoint_path, start_checkpoint), custom_objects={'Subpixel': Subpixel})
    else:
        generator = load_model("%s_%04d.h5" % (checkpoint_path, start_checkpoint), custom_objects={'Subpixel': Subpixel})
        discriminator = load_model("%s_dis_%04d.h5" % (checkpoint_path, start_checkpoint))


print("Load Training Data")
train_data = Dataset(src_path, 
                     0, train_config['train_data_count'], train_config['t_start'], train_config['t_end'], 
                     features, pre_config['var'], ref_path, [features[0]])

print("Source Data Shape: " + str(train_data.data[features[0]].shape))
print("Reference Data Shape: " + str(train_data.ref_data[features[0]].shape))

class NthLogger(keras.callbacks.Callback):
    def __init__(self,li=10,cpi=100,cpt_path="model", offset=0):
        self.act = offset
        self.li = li
        self.cpi = cpi
        self.cpt_path = cpt_path

    def on_epoch_end(self,batch,logs={}):
        self.act += 1
        if self.act % self.li == 0 or self.act == 1:
            print('%d/%d - loss: %f val_loss: %f' % (self.act, self.params['epochs'], logs['loss'], logs['val_loss']))
        if self.act % self.cpi == 0:
            path = "%s_%04d.h5" % (self.cpt_path, self.act//self.cpi)
            model.save(path)
            print('Saved Checkpoint: %s' % path)


print("Start Training")
if not train_config["adv_fac"] > 0.:
    x, y = train_data.get_data_splitted()

    history = model.fit(x=x,y=y, validation_split=train_config['val_split'], 
                        epochs=train_config['epochs'] - start_checkpoint*checkpoint_intervall, batch_size=train_config['batch_size'], 
                        verbose=0, callbacks=[NthLogger(log_intervall, checkpoint_intervall, checkpoint_path, start_checkpoint*checkpoint_intervall)])

    m_p = "%s_trained.h5" % model_path
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
else:
    # GAN
    z = Input(shape=(pre_config['patch_size'], pre_config['patch_size'],1), name='main')
    z_aux = Input(shape=(pre_config['patch_size'], pre_config['patch_size'],feature_cnt-1), name='aux')

    img = generator([z,z_aux])

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model  (stacked generator and discriminator)
    combined = Model([z,z_aux], [img,valid])
    combined.compile(loss=['mse','binary_crossentropy'], optimizer=keras.optimizers.adam(lr=train_config['learning_rate']),
                    loss_weights=[train_config['mse_fac'], train_config['adv_fac']])

    if verbose:
        combined.summary

    batch_size = train_config['batch_size']
    half_batch = train_config['batch_size']//2

    train_cnt = int(len(train_data.data[features[0]])*(1-train_config['val_split']))//batch_size*batch_size
    print('train count: %d' % train_cnt)
    eval_cnt = int(len(train_data.data[features[0]])*train_config['val_split'])//batch_size*batch_size
    print('eval count: %d' % eval_cnt)

    cnt_inv = batch_size/train_cnt

    history = {'d_loss':[],'d_acc':[],'g_loss':[],'g_mse':[],'g_adv_loss':[],
            'd_val_loss':[],'d_val_acc':[],'g_val_loss':[],'g_val_mse':[],'g_val_adv_loss':[]}
    idx0 = np.arange(train_cnt+eval_cnt)
    idx1 = np.arange(train_cnt+eval_cnt)

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
        
    idx0, val_idx0 = np.split(idx0,[train_cnt])
    idx1, val_idx1 = np.split(idx1,[train_cnt])

    for ep in range(start_checkpoint*checkpoint_intervall,train_config['epochs']):    
        # train
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        g_loss = [0.,0.,0.]
        d_loss = [0.,0.]
        
        for i in range(0,train_cnt,batch_size):
            x = train_data.get_data_splitted(idx0[i:i+half_batch])[0]
            y = train_data.get_data_splitted(idx0[i+half_batch:i+batch_size])[1]
            x = generator.predict(x)

            d_loss_fake = discriminator.train_on_batch(x, np.zeros((half_batch, 1)))
            d_loss_real = discriminator.train_on_batch(y, np.ones((half_batch, 1)))
            d_loss = np.add(d_loss, cnt_inv * 0.5 * np.add(d_loss_real, d_loss_fake) )
            
            x, y = train_data.get_data_splitted(idx1[i:i+batch_size])
            g_loss = np.add(g_loss, cnt_inv * np.array(combined.train_on_batch(x, [y[0],np.ones((batch_size, 1))])))
        
        # eval
        np.random.shuffle(val_idx0)
        np.random.shuffle(val_idx1)
        g_val_loss = [0.,0.,0.]
        d_val_loss = [0.,0.]
        
        x, y = train_data.get_data_splitted(val_idx0)
        x = generator.predict(x)
        
        d_loss_fake = discriminator.evaluate(x, np.zeros((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_loss_real = discriminator.evaluate(y, np.ones((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_val_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        x, y = train_data.get_data_splitted(val_idx1)
        g_val_loss = combined.evaluate(x, [y[0],np.ones((eval_cnt, 1))], batch_size=batch_size, verbose=0)
        
        history['d_loss'].append(d_loss[0])
        history['d_acc'].append(d_loss[1])
        history['d_val_loss'].append(d_val_loss[0])
        history['d_val_acc'].append(d_val_loss[1])
        history['g_loss'].append(g_loss[0])
        history['g_mse'].append(g_loss[1])
        history['g_adv_loss'].append(g_loss[2])
        history['g_val_loss'].append(g_val_loss[0])
        history['g_val_mse'].append(g_val_loss[1])
        history['g_val_adv_loss'].append(g_val_loss[2])
                
        if (ep+1) % log_intervall == 0 or ep == 0:
            print ("epoch %i" % (ep+1))
            print ("\ttrain: [D loss: %f, acc.: %.2f%%] [G loss: %f, mse: %f, adv: %f]" % (d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], g_loss[2]))
            print ("\teval.: [D loss: %f, acc.: %.2f%%] [G loss: %f, mse: %f, adv: %f]" % (d_val_loss[0], 100*d_val_loss[1], g_val_loss[0], g_val_loss[1], g_val_loss[2]))
        
        if (ep+1) % checkpoint_intervall == 0:
            path = "%s_%04d.h5" % (checkpoint_path, (ep+1)//checkpoint_intervall)
            generator.save(path)
            print('Saved Generator Checkpoint: %s' % path)
            path = "%s_%04d_dis.h5" % (checkpoint_path, (ep+1)//checkpoint_intervall)
            discriminator.save(path)
            print('Saved Generator Checkpoint: %s' % path)

    gen_p = "%s_trained.h5" % model_path
    generator.save(gen_p)
    print("Saved Model: %s" % gen_p)

    plt.plot(history['g_loss'])
    plt.plot(history['g_mse'])
    plt.plot(history['g_adv_loss'])
    plt.plot(history['g_val_mse'])
    plt.plot(history['g_val_loss'])
    plt.plot(history['g_val_adv_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['g_loss','g_mse','g_adv_loss','g_val_loss','g_val_mse','g_val_adv_loss'], loc='upper left')

    plt.savefig(fig_path+".png")
    plt.savefig(fig_path+".pdf")

    plt.clf()

    plt.plot(history['d_loss'])
    plt.plot(history['d_val_loss'])
    plt.title('discriminator loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['d_loss','d_val_loss'], loc='upper left')

    plt.savefig(fig_path+"_dis.png")
    plt.savefig(fig_path+"_dis.pdf")