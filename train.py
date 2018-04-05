import sys, os
sys.path.append("manta/scenes/tools/")
sys.path.append("hungarian/")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
from helpers import *

import math

#from dataset import Dataset
from gen_patches import gen_patches

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, ZeroPadding2D, Dense, MaxPooling2D
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, Activation, Flatten, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU
from subpixel import *
from spatial_transformer import *
from split_layer import *
from keras import regularizers
import numpy as np

paramUsed = []

data_path = getParam("data", "data/", paramUsed)
manta_path = getParam("manta", "manta/", paramUsed)
config_path = getParam("config", "config/version_00.txt", paramUsed)
verbose = int(getParam("verbose", 0, paramUsed)) != 0
gpu = getParam("gpu", "", paramUsed)

log_intervall = int(getParam("log_intervall", 1, paramUsed))
checkpoint_intervall = int(getParam("checkpoint_intervall", 1, paramUsed))

start_checkpoint = int(getParam("start_checkpoint", 0, paramUsed))

checkUnusedParam(paramUsed)

src_path = data_path + "patches/source/"
ref_path = data_path + "patches/reference/"

model_path = data_path + "models/"
if not os.path.exists(model_path):
	os.makedirs(model_path)

checkpoint_path = model_path + "checkpoints/"
if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)

if not gpu is "":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

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

patch_size = pre_config['patch_size']
ref_patch_size = pre_config['patch_size_ref']

model_path = '%s%s_%s' % (model_path, data_config['prefix'], config['id'])
checkpoint_path = '%s%s_%s' % (checkpoint_path, data_config['prefix'], config['id'])
print(model_path)
fig_path = '%s_loss' % model_path

src_path = "%s%s_%s-%s" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d"
ref_path = "%s%s_%s-%s" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d"
print(src_path)
print(ref_path)

loss_mode = train_config['loss']

particle_loss = keras.losses.mse

if loss_mode == 'hungarian_loss':
    from hungarian_loss import hungarian_loss
    particle_loss = hungarian_loss
elif loss_mode == 'emd_loss':
    from tf_approxmatch import emd_loss
    particle_loss = emd_loss
elif loss_mode == 'chamfer_loss':
    from tf_nndistance import chamfer_loss
    particle_loss = chamfer_loss

if start_checkpoint == 0:
    print("Generate Network")
    if train_config['explicit']:
        fac = 16
        
        k = train_config['par_feature_cnt']
        dropout = train_config['dropout']
        batch_size = train_config['batch_size']
        epochs = train_config['epochs']
        pre_train_stn = train_config['pre_train_stn']

        particle_cnt_src = pre_config['par_cnt']
        particle_cnt_dst = pre_config['par_cnt_ref']

        inputs = Input((particle_cnt_src,3), name="main")
        #aux_input = Input((particle_cnt_src,3))

        stn_input = Input((particle_cnt_src,3))
        stn = SpatialTransformer(stn_input,particle_cnt_src,dropout=dropout,quat=True,norm=True)
        stn_model = Model(inputs=stn_input, outputs=stn, name="stn")
        stn = stn_model(inputs)

        transformed = stn_transform(stn,inputs,quat=True, name="trans")

        #x = concatenate([intermediate, stn([x,aux_input])],axis=-1)

        x = [(Lambda(lambda v: v[:,i:i+1,:])(transformed)) for i in range(particle_cnt_src)]

        x = split_layer(Dropout(dropout),x)
        x = split_layer(Dense(fac, activation='tanh'),x)
        x = split_layer(Dropout(dropout),x)
        x = split_layer(Dense(fac, activation='tanh'),x)

        x = concatenate(x, axis=1)
        x = stn_transform(SpatialTransformer(x,particle_cnt_src,fac,1),x)

        x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_src)]

        x = split_layer(Dropout(dropout),x)
        x = split_layer(Dense(fac, activation='tanh'),x)
        x = split_layer(Dropout(dropout),x)
        x = split_layer(Dense(fac*2, activation='tanh'),x)
        x = split_layer(Dropout(dropout),x)
        x = split_layer(Dense(k, activation='tanh'),x)

        x = add(x)

        x = Flatten()(x)
        x = Dropout(dropout)(x)
        x = Dense(fac*8, activation='tanh')(x)
        x = Dropout(dropout)(x)
        x = Dense(fac*4, activation='tanh')(x)
        x = Dropout(dropout)(x)
        x = Dense(3*particle_cnt_dst, activation='tanh')(x)

        inv_par_out = Reshape((particle_cnt_dst,3))(x)
        out = stn_transform_inv(stn,inv_par_out,quat=True)

        if feature_cnt > 1:
            auxiliary_input = Input(shape=(patch_size, patch_size, feature_cnt-1), name="auxiliary_input")  

        model = Model(inputs=[inputs, auxiliary_input], outputs=out) if feature_cnt > 1 else Model(inputs=[inputs], outputs=out)
        model.compile(loss=particle_loss, optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]))

        model.save(model_path + '.h5')
        
        if verbose: 
            model.summary()
            
        if train_config["adv_fac"] > 0.:
            generator = model

            disc_input = Input((particle_cnt_dst,3))

            disc_stn_input = Input((particle_cnt_dst,3))
            disc_stn = SpatialTransformer(disc_stn_input,particle_cnt_dst,dropout=dropout,quat=True,norm=True)
            disc_stn_model = Model(inputs=disc_stn_input, outputs=disc_stn, name="disc_stn")
            disc_stn = disc_stn_model(disc_input)

            disc_transformed = stn_transform(disc_stn,disc_input,quat=True, name="disc_trans")

            x = [(Lambda(lambda v: v[:,i:i+1,:])(disc_transformed)) for i in range(particle_cnt_dst)]

            x = split_layer(Dropout(dropout),x)
            x = split_layer(Dense(fac, activation='tanh'),x)
            x = split_layer(Dropout(dropout),x)
            x = split_layer(Dense(fac, activation='tanh'),x)

            x = concatenate(x, axis=1)
            x = stn_transform(SpatialTransformer(x,particle_cnt_dst,fac,1),x)

            x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_dst)]

            x = split_layer(Dropout(dropout),x)
            x = split_layer(Dense(fac, activation='tanh'),x)
            x = split_layer(Dropout(dropout),x)
            x = split_layer(Dense(fac*2, activation='tanh'),x)
            x = split_layer(Dropout(dropout),x)
            x = split_layer(Dense(k, activation='tanh'),x)

            x = add(x)

            x = Flatten()(x)
            x = Dropout(dropout)(x)
            x = Dense(fac*16, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(fac*8, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(fac*4, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(1, activation='sigmoid')(x)

            '''disc_input = Input((particle_cnt_src,3), name="main")
            x = Flatten()(disc_input)
            x = Dropout(dropout)(x)
            x = Dense(fac*32, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(fac*16, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(fac*8, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(fac*4, activation='tanh')(x)
            x = Dropout(dropout)(x)
            x = Dense(1, activation='sigmoid')(x)'''

            discriminator = Model(inputs=disc_input, outputs=x)
            discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]), metrics=['accuracy'])

            discriminator.save(model_path + '_dis.h5')

            if verbose: 
                discriminator.summary()
    else:   
        inputs = Input((patch_size, patch_size, 1), name="main_input")
        auxiliary_input = Input(shape=(patch_size, patch_size, feature_cnt-1), name="auxiliary_input")

        base = Reshape((patch_size*patch_size,), name="reshape_flat")(inputs)
        base = RepeatVector(9, name="repeate")(base)
        base = Permute((2, 1), name="permute")(base)
        base = Reshape((patch_size, patch_size,9), name="reshape_back")(base)
        
        x = concatenate([inputs, auxiliary_input], name="concatenate")
        x = Reshape((patch_size*patch_size*feature_cnt,), name="reshape_flat_res")(x)
        x = RepeatVector(9, name="repeate_res")(x)
        x = Permute((2, 1), name="permute_res")(x)
        x = Reshape((patch_size, patch_size,9*feature_cnt), name="reshape_back_res")(x)
        
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

            img_shape = (ref_patch_size, ref_patch_size, 1)

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
    if train_config["adv_fac"] <= 0.:
        model = load_model("%s_%04d.h5" % (checkpoint_path, start_checkpoint), custom_objects={'Subpixel': Subpixel, loss_mode: particle_loss})
    else:
        generator = load_model("%s_%04d.h5" % (checkpoint_path, start_checkpoint), custom_objects={'Subpixel': Subpixel, loss_mode: particle_loss})
        model = generator
        discriminator = load_model("%s_dis_%04d.h5" % (checkpoint_path, start_checkpoint))


print("Load Training Data")
src_data, ref_data, src_rot_data, ref_rot_data = gen_patches(data_path, config_path, 
    int(data_config['data_count']*train_config['train_split']), train_config['t_end'], 
    pre_config['var'], pre_config['par_var'], t_start=train_config['t_start'])[:4]
'''train_data = Dataset(src_path, 
                     0, int(data_config['data_count']*train_config['train_split']), train_config['t_start'], train_config['t_end'], 
                     features, pre_config['var'], pre_config['par_var'], ref_path, [features[0]])'''

#print("Source Data Shape: " + str(train_data.data[features[0]].shape))
#print("Reference Data Shape: " + str(train_data.ref_data[features[0]].shape))

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

if pre_train_stn:
    inputs = model.inputs[0]
    stn = model.get_layer("stn")
    trans_input = model.get_layer("trans")([inputs, stn(inputs)])
    inter_model = Model(inputs=inputs, outputs=trans_input)
    inter_model.compile(loss=particle_loss, optimizer=keras.optimizers.adam(lr=train_config['learning_rate']))
    inter_model.fit(x=src_data,y=src_rot_data,epochs=train_config['epochs'],batch_size=train_config['batch_size'])
    stn.trainable = False        
    model.compile(loss=particle_loss, optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]))
    if train_config["adv_fac"] > 0:
        inputs = discriminator.inputs[0]
        stn = model.get_layer("disc_stn")
        trans_input = model.get_layer("disc_trans")([inputs, stn(inputs)])
        inter_model = Model(inputs=inputs, outputs=trans_input)
        inter_model.compile(loss=particle_loss, optimizer=keras.optimizers.adam(lr=train_config['learning_rate']))
        inter_model.fit(x=ref_data,y=ref_rot_data,epochs=train_config['epochs'],batch_size=train_config['batch_size'])
        stn.trainable = False        

if train_config["adv_fac"] <= 0.:
    val_split = train_config['val_split']

    history = model.fit(x=src_data,y=ref_data, validation_split=val_split, 
                        epochs=train_config['epochs'] - start_checkpoint*checkpoint_intervall, batch_size=train_config['batch_size'], 
                        verbose=1, callbacks=[NthLogger(log_intervall, checkpoint_intervall, checkpoint_path, start_checkpoint*checkpoint_intervall)])

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
    z = [Input(shape=(particle_cnt_src,3) if train_config['explicit'] else (patch_size, patch_size,1), name='main')]
    if feature_cnt > 1:
        z = np.append(z, [Input(shape=(patch_size, patch_size,feature_cnt-1), name='aux')])

    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model  (stacked generator and discriminator)
    combined = Model(z, [img,valid])
    combined.compile(loss=[generator.loss_functions[0],discriminator.loss_functions[0]], optimizer=keras.optimizers.adam(lr=train_config['learning_rate']),
                    loss_weights=[train_config['mse_fac'], train_config['adv_fac']])

    if verbose:
        combined.summary

    batch_size = train_config['batch_size']
    half_batch = train_config['batch_size']//2

    train_cnt = int(len(src_data)*(1-train_config['val_split']))//batch_size*batch_size
    print('train count: %d' % train_cnt)
    eval_cnt = int(len(src_data)*train_config['val_split'])//batch_size*batch_size
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
            print("Train epoch {}, batch {}/{}".format(ep+1, i+batch_size, train_cnt), end="\r", flush=True)
            x = src_data[idx0[i:i+half_batch]]
            y = ref_data[idx0[i+half_batch:i+batch_size]]
            x = generator.predict(x)

            d_loss_fake = discriminator.train_on_batch(x, np.zeros((half_batch, 1)))
            d_loss_real = discriminator.train_on_batch(y, np.ones((half_batch, 1)))
            d_loss = np.add(d_loss, cnt_inv * 0.5 * np.add(d_loss_real, d_loss_fake) )
            
            x = src_data[idx1[i:i+batch_size]]
            y = ref_data[idx1[i:i+batch_size]]
            g_loss = np.add(g_loss, cnt_inv * np.array(combined.train_on_batch(x, [y, np.ones((batch_size, 1))])))
        
        print("\r", flush=True)
        # eval
        np.random.shuffle(val_idx0)
        np.random.shuffle(val_idx1)
        g_val_loss = [0.,0.,0.]
        d_val_loss = [0.,0.]
    
        x = src_data[val_idx0]
        y = ref_data[val_idx0]
        x = generator.predict(x)
        
        d_loss_fake = discriminator.evaluate(x, np.zeros((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_loss_real = discriminator.evaluate(y, np.ones((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_val_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        x = src_data[val_idx1]
        y = ref_data[val_idx1]
        g_val_loss = combined.evaluate(x, [y,np.ones((eval_cnt, 1))], batch_size=batch_size, verbose=0)
        
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