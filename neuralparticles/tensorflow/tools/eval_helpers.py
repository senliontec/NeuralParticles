import keras
import tensorflow as tf

from neuralparticles.tools.plot_helpers import plot_particles, write_csv
from neuralparticles.tools.data_helpers import PatchExtractor

import io, os

def eval_patch(model, src, path="", ref=None, features=[], z=None, verbose=0):
    result = model.predict(src)

    if type(result) is list:
        if verbose > 0: print("Truncate points at %d" % int(result[1][0] * result[0].shape[1]))
        result = result[0][0,:int(result[1][0] * result[0].shape[1])]
    else:
        result = result[0]

    if path != "" and verbose > 0:
        _i = 0
        vel_src = None
        for i in range(1,len(features)):
            if features[i] == 'v':
                aux_src = src[1][:,_i:_i+3]
                vel_src = aux_src/100
                _i += 3
            else:
                aux_src = src[1][:,_i:_i+1]
                _i += 1
            if verbose > 2: write_csv(path + "_%s.csv"%features[i], aux_src)
        

        if verbose > 1: plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path + ".svg", ref=ref, src=src[0][0], vel=vel_src, z = z)
        if verbose > 0: plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path + ".png", ref=ref, src=src[0][0], vel=vel_src, z = z)
        if verbose > 2: write_csv(path + "_res.csv", result)
        if verbose > 2: write_csv(path + "_ref.csv", ref)
        if verbose > 2: write_csv(path + "_src.csv", src[0][0])

    return result

def eval_frame(model, patch_extractor, factor_d, path="", src=None, aux=None, ref=None, hdim=0, z=None, verbose=0):
    result = model.predict(x=patch_extractor.get_patches())
    if type(result) is list:
        for i in range(len(patch_extractor.positions)):
            patch_extractor.set_patch(result[0][i,:int(result[1][i] * result[0].shape[1])], i)
    else:
        patch_extractor.set_patches(result)

    '''while(True):
        s = patch_extractor.get_patch()
        if s is None:
            break
        result = model.predict(x=s)
        if type(result) is list:
            result = result[0][0,:int(result[1][0] * result[0].shape[1])]
        else:
            result = result[0]
        patch_extractor.set_patch(result)'''
    result = patch_extractor.data * factor_d
    if path != "" and verbose > 0:
        vel_src = None
        for k in aux:
            aux_src = aux[k]
            if k == 'v':
                vel_src = aux_src/100
            if verbose > 2: write_csv(path + "_%s.csv"%k, aux_src)

        if verbose > 1: plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path + ".svg", ref=ref, src=src*factor_d, vel=vel_src, z = z)
        if verbose > 0: plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path + ".png", ref=ref, src=src*factor_d, vel=vel_src, z = z)
        if verbose > 2: write_csv(path + "_res.csv", result)
        if verbose > 2: write_csv(path + "_ref.csv", ref)
        if verbose > 2: write_csv(path + "_src.csv", src)
    patch_extractor.reset()
    return result

def add_images(writer, tag, src, ref, res, epoch, xlim=None, ylim=None, s=1.0, z=None):
    imgs = []
    buf = io.BytesIO()
    plot_particles(res, path=buf, src=src, ref=ref, xlim=xlim, ylim=ylim, s=s, z = z)
    img = tf.Summary.Image(colorspace=4, encoded_image_string=buf.getvalue())
    imgs.append(tf.Summary.Value(tag=tag+"_comp", image=img))
    buf = io.BytesIO()
    plot_particles(res, path=buf, xlim=xlim, ylim=ylim, s=s, z = z)
    img = tf.Summary.Image(colorspace=4, encoded_image_string=buf.getvalue())
    imgs.append(tf.Summary.Value(tag=tag+"_res", image=img))
    buf = io.BytesIO()
    plot_particles(src, path=buf, xlim=xlim, ylim=ylim, s=s, z = z)
    img = tf.Summary.Image(colorspace=4, encoded_image_string=buf.getvalue())
    imgs.append(tf.Summary.Value(tag=tag+"_src", image=img))
    buf = io.BytesIO()
    plot_particles(ref, path=buf, xlim=xlim, ylim=ylim, s=s, z = z)
    img = tf.Summary.Image(colorspace=4, encoded_image_string=buf.getvalue())
    imgs.append(tf.Summary.Value(tag=tag+"_ref", image=img))

    writer.add_summary(tf.Summary(value=imgs), epoch)
    writer.flush()

    return

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
            self.model.save(path)
            print('Saved Checkpoint: %s' % path)

class EvalCallback(keras.callbacks.TensorBoard):
    def __init__(self, path, src, ref, 
                 features=[], 
                 batch_intervall=0, 
                 z=None, verbose=0,
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        
        super().__init__(os.path.dirname(path) + "/logs/patch/",
                         histogram_freq,
                         batch_size,
                         write_graph,
                         write_grads,
                         write_images,
                         embeddings_freq,
                         embeddings_layer_names,
                         embeddings_metadata)
        #super().set_model(model)
        self.path = path + "_%03d_%03d"
        self.tag = os.path.basename(path) + "_%03d"
        self.src = src
        self.ref = ref
        self.features = features
        self.batch_intervall = batch_intervall
        self.z = z
        self.verbose = verbose

    '''def on_train_begin(self, logs={}):
        for i in range(len(self.src)):
            res = eval_patch(self.model, self.src[i], self.path%(i,0), self.ref[i], self.features, self.z, self.verbose)
            add_images(self.writer, self.tag%i, self.src[i][0][0], self.ref[i], res, 0, xlim=[-1,1], ylim=[-1,1], s=5)'''

    
    def on_epoch_end(self,ep,logs={}):
        super().on_epoch_end(ep, logs)
        if self.batch_intervall > 0:
            return
        print("Eval Patch")
        for i in range(len(self.src)):
            res = eval_patch(self.model, self.src[i], self.path%(i,ep), self.ref[i], self.features, self.z, self.verbose)
            add_images(self.writer, self.tag%i, self.src[i][0][0], self.ref[i], res, ep, xlim=[-1,1], ylim=[-1,1], s=5, z=self.z)

    def on_batch_end(self,batch,logs={}):
        if self.batch_intervall <= 0 or batch % self.batch_intervall != 0 or batch > 10000:
            return
        for i in range(len(self.src)):
            res = eval_patch(self.model, self.src[i], self.path%(i,batch//self.batch_intervall), self.ref[i], self.features, self.z, self.verbose)
            add_images(self.writer, self.tag%i, self.src[i][0][0], self.ref[i], res, batch//self.batch_intervall, xlim=[-1,1], ylim=[-1,1], s=5, z=self.z)

class EvalCompleteCallback(keras.callbacks.TensorBoard):
    def __init__(self, path, patch_extractor, ref, factor_d, hdim, 
                 batch_intervall=0, 
                 z=None, verbose=0,
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        
        super().__init__(os.path.dirname(path) + "/logs/complete/",
                         histogram_freq,
                         batch_size,
                         write_graph,
                         write_grads,
                         write_images,
                         embeddings_freq,
                         embeddings_layer_names,
                         embeddings_metadata)
        #super().set_model(model)
        self.path = path + "_%03d_%03d"
        self.tag = os.path.basename(path) + "_%03d"
        self.patch_extractor = patch_extractor
        self.ref = ref
        self.factor_d = factor_d
        self.hdim = hdim
        self.batch_intervall = batch_intervall
        self.z = z
        self.verbose = verbose
    
    '''def on_train_begin(self, logs={}):
        for i in range(len(self.patch_extractor)):
            res = eval_frame(self.model, self.patch_extractor[i], self.factor_d, self.path%(i,0), self.patch_extractor[i].src_data, self.patch_extractor[i].aux_data, self.ref[i], self.hdim, self.z, verbose=self.verbose)
            add_images(self.writer, self.tag%i, self.patch_extractor[i].src_data * self.factor_d, self.ref[i], res, 0, xlim=[0,self.hdim], ylim=[0,self.hdim], s=0.1)'''

    def on_epoch_end(self,ep,logs={}):
        super().on_epoch_end(ep, logs)
        if self.batch_intervall > 0:
            return
        print("Eval")
        for i in range(len(self.patch_extractor)):
            res = eval_frame(self.model, self.patch_extractor[i], self.factor_d, self.path%(i,ep), self.patch_extractor[i].src_data, self.patch_extractor[i].aux_data, self.ref[i], self.hdim, self.z, verbose=self.verbose)
            add_images(self.writer, self.tag%i, self.patch_extractor[i].src_data * self.factor_d, self.ref[i], res, ep, xlim=[0,self.hdim], ylim=[0,self.hdim], s=0.1, z=self.z)

    def on_batch_end(self,batch,logs={}):
        if self.batch_intervall <= 0 or batch % self.batch_intervall != 0:
            return
        for i in range(len(self.patch_extractor)):
            res = eval_frame(self.model, self.patch_extractor[i], self.factor_d, self.path%(i,batch//self.batch_intervall), self.patch_extractor[i].src_data, self.patch_extractor[i].aux_data, self.ref[i], self.hdim, self.z, verbose=self.verbose)
            add_images(self.writer, self.tag%i, self.patch_extractor[i].src_data * self.factor_d, self.ref[i], res, batch//self.batch_intervall, xlim=[0,self.hdim], ylim=[0,self.hdim], s=0.1, z=self.z)