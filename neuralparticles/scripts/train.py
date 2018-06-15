import os

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

import math

import tensorflow as tf

from neuralparticles.tools.param_helpers import *
from neuralparticles.tools.data_helpers import load_patches_from_file, PatchExtractor, get_data_pair, extract_particles

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, ZeroPadding2D, Dense, MaxPooling2D
from keras.layers import Reshape, RepeatVector, Permute, concatenate, add, average, Activation, Flatten, Lambda, Dropout, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers

from neuralparticles.tensorflow.tools.spatial_transformer import *
from neuralparticles.tensorflow.tools.zero_mask import zero_mask, trunc_mask

from neuralparticles.tensorflow.tools.eval_helpers import *

import numpy as np

data_path = getParam("data", "data/")
config_path = getParam("config", "config/version_00.txt")
verbose = int(getParam("verbose", 0)) != 0
gpu = getParam("gpu", "")

eval_dataset = int(getParam("eval_d", 18))
eval_t = int(getParam("eval_t", 10))
eval_var = int(getParam("eval_v", 0))
eval_patch_idx = int(getParam("eval_i", 8))

log_interval = int(getParam("log_interval", 1))
checkpoint_interval = int(getParam("checkpoint_interval", 1))

start_checkpoint = int(getParam("start_checkpoint", 0))

checkUnusedParams()

src_path = data_path + "patches/source/"
ref_path = data_path + "patches/reference/"

model_path = data_path + "models/"
if not os.path.exists(model_path):
	os.mkdir(model_path)

checkpoint_path = model_path + "checkpoints/"
if not os.path.exists(checkpoint_path):
	os.mkdir(checkpoint_path)

tmp_folder = backupSources(data_path)
tmp_model_path = tmp_folder + "models/"
os.mkdir(tmp_model_path)
tmp_checkpoint_path = tmp_model_path + "checkpoints/"
os.mkdir(tmp_checkpoint_path)
tmp_eval_path = tmp_folder + "eval/"
os.mkdir(tmp_eval_path)

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

# copy config files into tmp

np.random.seed(data_config['seed'])
tf.set_random_seed(data_config['seed'])

features = train_config['features'][1:]
feature_cnt = len(features)
if 'v' in features:
    feature_cnt += 2
print("feature_count: %d" % feature_cnt)

patch_size = pre_config['patch_size']
ref_patch_size = pre_config['patch_size_ref']
par_cnt = pre_config['par_cnt']
ref_par_cnt = pre_config['par_cnt_ref']

factor_2D = math.sqrt(pre_config['factor'])

pad_val = -2.0

hdim = data_config['res']
dim = int(hdim/factor_2D)

tmp_model_path = '%s%s_%s' % (tmp_model_path, data_config['prefix'], config['id'])
tmp_checkpoint_path = '%s%s_%s' % (tmp_checkpoint_path, data_config['prefix'], config['id'])
print(tmp_model_path)
fig_path = '%s_loss' % tmp_model_path

src_path = "%s%s_%s-%s" % (src_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d"
ref_path = "%s%s_%s-%s" % (ref_path, data_config['prefix'], data_config['id'], pre_config['id']) + "_d%03d_var%02d_pvar%02d_%03d"
print(src_path)
print(ref_path)

loss_mode = train_config['loss']

particle_loss = keras.losses.mse

if loss_mode == 'hungarian_loss':
    from neuralparticles.tensorflow.losses.hungarian_loss import hungarian_loss
    particle_loss = hungarian_loss
elif loss_mode == 'emd_loss':
    from neuralparticles.tensorflow.losses.tf_approxmatch import emd_loss
    particle_loss = emd_loss
elif loss_mode == 'chamfer_loss':
    from neuralparticles.tensorflow.losses.tf_nndistance import chamfer_loss
    particle_loss = chamfer_loss
else:
    print("No matching loss specified! Fallback to MSE loss.")

def mask_loss(y_true, y_pred):
    return particle_loss(y_true * zero_mask(y_true, pad_val), y_pred)

if start_checkpoint == 0:
    print("Generate Network")
    
    fac = 16
        
    k = train_config['par_feature_cnt']
    dropout = train_config['dropout']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    pre_train_stn = train_config['pre_train_stn']
    learning_rate = train_config['learning_rate']

    particle_cnt_src = pre_config['par_cnt']
    particle_cnt_dst = pre_config['par_cnt_ref']

    inputs = [Input((particle_cnt_src,3), name='main_input')]

    mask = Lambda(lambda v: zero_mask(v, pad_val))(inputs[0])

    x = Multiply()([inputs[0], mask])

    if feature_cnt > 0:
        aux_input = Input((particle_cnt_src, feature_cnt), name='aux_input')
        inputs.append(aux_input)
        aux_input = Multiply()([aux_input, mask])

    stn_input = Input((particle_cnt_src,3))
    stn = SpatialTransformer(stn_input,particle_cnt_src,dropout=dropout,quat=True,norm=True)
    stn_model = Model(inputs=stn_input, outputs=stn, name="stn")
    stn = stn_model(x)

    transformed = stn_transform(stn,x,quat=True, name='trans')
    
    if feature_cnt > 0:
        if 'v' in features:
            aux_input = Lambda(lambda a: concatenate([stn_transform(stn, a[:,:,:3]/100,quat=True),a[:,:,3:]], axis=-1), name='aux_trans')(aux_input)
        transformed = concatenate([transformed, aux_input], axis=-1)

    x = [Lambda(lambda v: v[:,i,:])(transformed) for i in range(particle_cnt_src)]

    x = list(map(Dropout(dropout),x))
    x = list(map(Dense(fac, activation='tanh'),x))
    x = list(map(Dropout(dropout),x))
    x = list(map(Dense(fac, activation='tanh'),x))

    x = list(map(Lambda(lambda v: K.expand_dims(v, axis=1)),x))
    x = concatenate(x, axis=1)

    x = stn_transform(SpatialTransformer(x,particle_cnt_src,fac,1),x)

    x = [Lambda(lambda v: v[:,i,:])(x) for i in range(particle_cnt_src)]

    x = list(map(Dropout(dropout),x))
    x = list(map(Dense(fac, activation='tanh'),x))
    x = list(map(Dropout(dropout),x))
    x = list(map(Dense(fac*2, activation='tanh'),x))
    x = list(map(Dropout(dropout),x))
    x = list(map(Dense(k, activation='tanh'),x))
    
    x = [Lambda(lambda v: v * mask[:,i])(x[i]) for i in range(particle_cnt_src)]

    latent = add(x)

    x = Dropout(dropout)(latent)
    x = Dense(fac)(x)
    x = Dropout(dropout)(x)
    trunc = Dense(1)(x)
    out_mask = Lambda(lambda v: trunc_mask(v,particle_cnt_dst))(trunc)

    x = Dropout(dropout)(latent)
    x = Dense(particle_cnt_dst, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(particle_cnt_dst, activation='tanh')(x)
    x = Dropout(dropout)(x)
    x = Dense(3*particle_cnt_dst, activation='tanh')(x)

    inv_par_out = Reshape((particle_cnt_dst,3))(x)
    out = stn_transform_inv(stn,inv_par_out,quat=True)

    #out = Multiply()([out, Reshape((particle_cnt_dst,1))(out_mask)])

    model = Model(inputs=inputs, outputs=[out,trunc])
    model.compile(loss=[mask_loss, 'mse'], optimizer=keras.optimizers.adam(lr=learning_rate), loss_weights=[1.0,0.0])

    model.save(tmp_model_path + '.h5')
    
    if verbose: 
        model.summary()
        stn_model.summary()
        
    if train_config["adv_fac"] > 0.:
        generator = model

        disc_input = Input((particle_cnt_dst,3), name='disc_input')

        disc_stn_input = Input((particle_cnt_dst,3))
        disc_stn = SpatialTransformer(disc_stn_input,particle_cnt_dst,dropout=dropout,quat=True,norm=True)
        disc_stn_model = Model(inputs=disc_stn_input, outputs=disc_stn, name="disc_stn")
        disc_stn = disc_stn_model(disc_input)

        disc_transformed = stn_transform(disc_stn,disc_input,quat=True, name="disc_trans")

        x = [(Lambda(lambda v: v[:,i:i+1,:])(disc_transformed)) for i in range(particle_cnt_dst)]

        x = list(map(Dropout(dropout),x))
        x = list(map(Dense(fac, activation='tanh'),x))
        x = list(map(Dropout(dropout),x))
        x = list(map(Dense(fac, activation='tanh'),x))

        x = concatenate(x, axis=1)
        x = stn_transform(SpatialTransformer(x,particle_cnt_dst,fac,1),x)

        x = [(Lambda(lambda v: v[:,i:i+1,:])(x)) for i in range(particle_cnt_dst)]

        x = list(map(Dropout(dropout),x))
        x = list(map(Dense(fac, activation='tanh'),x))
        x = list(map(Dropout(dropout),x))
        x = list(map(Dense(fac*2, activation='tanh'),x))
        x = list(map(Dropout(dropout),x))
        x = list(map(Dense(k, activation='tanh'),x))

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

        discriminator = Model(inputs=disc_input, outputs=x)
        discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=learning_rate), metrics=['accuracy'])

        discriminator.save(tmp_model_path + '_dis.h5')

        if verbose: 
            discriminator.summary()
else:
    if train_config["adv_fac"] <= 0.:
        model = load_model("%s_%04d.h5" % (checkpoint_path, start_checkpoint), custom_objects={loss_mode: mask_loss})
    else:
        generator = load_model("%s_%04d.h5" % (checkpoint_path, start_checkpoint), custom_objects={loss_mode: mask_loss})
        model = generator
        discriminator = load_model("%s_dis_%04d.h5" % (checkpoint_path, start_checkpoint))


print("Load Training Data")

src_data, ref_data, src_rot_data, ref_rot_data = load_patches_from_file(data_path, config_path)

idx = np.arange(src_data[0].shape[0])
np.random.shuffle(idx)
src_data = [s[idx] for s in src_data]
ref_data = ref_data[idx]
src_rot_data = src_rot_data[idx]
ref_rot_data = ref_rot_data[idx]

print("Load Eval Data")
(eval_src_data, eval_sdf_data, eval_par_aux), (eval_ref_data, eval_ref_sdf_data) = get_data_pair(data_path, config_path, eval_dataset, eval_t, eval_var) 
patch_extractor = PatchExtractor(eval_src_data, eval_sdf_data, patch_size, par_cnt, pre_config['surf'], pre_config['stride'], aux_data=eval_par_aux, features=features)
eval_src_patch = patch_extractor.get_patch_idx(eval_patch_idx)
eval_ref_patch = extract_particles(eval_ref_data, patch_extractor.positions[eval_patch_idx] * factor_2D, ref_par_cnt, ref_patch_size)[0]

print("Eval trunc src: %.02f" % (np.count_nonzero(eval_src_patch[0][:,:,:1] != pad_val)/particle_cnt_src))
print("Eval trunc ref: %.02f" % (np.count_nonzero(eval_ref_patch[:,:1] != pad_val)/particle_cnt_dst))

val_split = train_config['val_split']

print("Start Training")
if pre_train_stn and False:
    inputs = model.inputs[0]
    mask = Lambda(lambda v: zero_mask(v, pad_val))(inputs)
    x = Multiply()([inputs,mask])
    stn = model.get_layer("stn")
    trans_input = model.get_layer("trans")([x, stn(x)])
    inter_model = Model(inputs=inputs, outputs=trans_input)
    inter_model.compile(loss=mask_loss, optimizer=keras.optimizers.adam(lr=train_config['learning_rate']))
    history = inter_model.fit(x=src_data[0],y=src_rot_data,epochs=train_config['pre_train_epochs'],batch_size=train_config['batch_size'], validation_split=val_split,
        verbose=1, callbacks=[EvalCallback(tmp_eval_path + "inter_eval_patch_%03d", inter_model, eval_src_patch, eval_ref_patch)])
    stn.trainable = False        
    model.compile(loss=mask_loss, optimizer=keras.optimizers.adam(lr=train_config["learning_rate"]))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig(fig_path+"_stn.png")
    plt.savefig(fig_path+"_stn.pdf")

    if train_config["adv_fac"] > 0:
        inputs = discriminator.inputs[0]
        stn = model.get_layer("disc_stn")
        trans_input = model.get_layer("disc_trans")([inputs, stn(inputs)])
        inter_model = Model(inputs=inputs, outputs=trans_input)
        inter_model.compile(loss=mask_loss, optimizer=keras.optimizers.adam(lr=train_config['learning_rate']))
        inter_model.fit(x=ref_data,y=ref_rot_data,epochs=train_config['pre_train_epochs'],batch_size=train_config['batch_size'])
        stn.trainable = False  

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.savefig(fig_path+"_dis_stn.png")
        plt.savefig(fig_path+"_dis_stn.pdf")      

if train_config["adv_fac"] <= 0.:
    trunc_ref = np.count_nonzero(ref_data[:,:,:1] != pad_val, axis=1)/particle_cnt_dst
    history = model.fit(x=src_data,y=[ref_data, trunc_ref], validation_split=val_split, 
                        epochs=epochs - start_checkpoint*checkpoint_interval, batch_size=train_config['batch_size'], 
                        verbose=1)#, callbacks=[NthLogger(model, log_interval, checkpoint_interval, tmp_checkpoint_path, start_checkpoint*checkpoint_interval),
                                  #            EvalCallback(tmp_eval_path + "eval_patch_%03d", model, eval_src_patch, eval_ref_patch, features),
                                  #            EvalCompleteCallback(tmp_eval_path + "eval_%03d", model, patch_extractor, eval_ref_data, factor_2D, hdim)])

    m_p = "%s_trained.h5" % tmp_model_path
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
    z = [Input(shape=(particle_cnt_src,3), name='gan_input')]
    if feature_cnt > 0:
        z = np.append(z, [Input(shape=(particle_cnt_src,feature_cnt), name='gan_aux')])

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
    half_batch = batch_size//2

    train_cnt = int(len(src_data[0])*(1-train_config['val_split']))//batch_size*batch_size
    print('train count: %d' % train_cnt)
    eval_cnt = int(len(src_data[0])*train_config['val_split'])//batch_size*batch_size
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

    for ep in range(start_checkpoint*checkpoint_interval,epochs):    
        # train
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        g_loss = [0.,0.,0.]
        d_loss = [0.,0.]
        
        for i in range(0,train_cnt,batch_size):
            print("Train epoch {}, batch {}/{}".format(ep+1, i+batch_size, train_cnt), end="\r", flush=True)
            x = [s[idx0[i:i+half_batch]] for s in src_data]
            y = ref_data[idx0[i+half_batch:i+batch_size]]
            x = generator.predict(x)

            d_loss_fake = discriminator.train_on_batch(x, np.zeros((half_batch, 1)))
            d_loss_real = discriminator.train_on_batch(y, np.ones((half_batch, 1)))
            d_loss = np.add(d_loss, cnt_inv * 0.5 * np.add(d_loss_real, d_loss_fake) )
            
            x = [s[idx1[i:i+batch_size]] for s in src_data]
            y = ref_data[idx1[i:i+batch_size]]
            g_loss = np.add(g_loss, cnt_inv * np.array(combined.train_on_batch(x, [y, np.ones((batch_size, 1))])))
        
        print("\r", flush=True)
        # eval
        np.random.shuffle(val_idx0)
        np.random.shuffle(val_idx1)
        g_val_loss = [0.,0.,0.]
        d_val_loss = [0.,0.]
    
        x = [s[val_idx0] for s in src_data]
        y = ref_data[val_idx0]
        x = generator.predict(x)
        
        d_loss_fake = discriminator.evaluate(x, np.zeros((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_loss_real = discriminator.evaluate(y, np.ones((eval_cnt, 1)), batch_size=half_batch, verbose=0)
        d_val_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        x = [s[val_idx1] for s in src_data]
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
                
        if (ep+1) % log_interval == 0 or ep == 0:
            print ("epoch %i" % (ep+1))
            print ("\ttrain: [D loss: %f, acc.: %.2f%%] [G loss: %f, mse: %f, adv: %f]" % (d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], g_loss[2]))
            print ("\teval.: [D loss: %f, acc.: %.2f%%] [G loss: %f, mse: %f, adv: %f]" % (d_val_loss[0], 100*d_val_loss[1], g_val_loss[0], g_val_loss[1], g_val_loss[2]))
        
        if (ep+1) % checkpoint_interval == 0:
            path = "%s_%04d.h5" % (tmp_checkpoint_path, (ep+1)//checkpoint_interval)
            generator.save(path)
            print('Saved Generator Checkpoint: %s' % path)
            path = "%s_%04d_dis.h5" % (tmp_checkpoint_path, (ep+1)//checkpoint_interval)
            discriminator.save(path)
            print('Saved Generator Checkpoint: %s' % path)

        print("Eval")
        eval_patch(generator, eval_src_patch, tmp_eval_path + "eval_patch_%03d"%ep, eval_ref_patch, features)
        eval_frame(generator, patch_extractor, factor_2D, "eval_%03d"%ep, patch_extractor.src_data, patch_extractor.aux_data, eval_ref_data, hdim)

    gen_p = "%s_trained.h5" % tmp_model_path
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

while(True):
    char = input("\nTrained Model only saved temporarily, do you want to save it? [y/n]\n")
    if char == "y" or char == "Y":
        from distutils.dir_util import copy_tree
        copy_tree(os.path.dirname(tmp_model_path), model_path)
        break
    elif char == "n" or char == "N":
        break
