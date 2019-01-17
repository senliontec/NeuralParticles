import keras
import tensorflow as tf
import numpy as np

from neuralparticles.tools.plot_helpers import plot_particles, write_csv
from neuralparticles.tools.data_helpers import PatchExtractor, cluster_analysis

import io, os

def eval_patch(model, src, path="", ref=None, features=[], z=None, verbose=0, truncate=False):
    result = model.predict(src)

    if type(result) is list:
        raw_result = result[0][0]
        cnt = int(result[1][0] * result[0].shape[1])
        if verbose > 0: print("Reduce points to: " + str(cnt))
        result = result[0][0,:cnt]
    elif truncate:
        raw_result = result[0]
        cnt = int(np.count_nonzero(src[0][...,1] != -2.0)) * (result.shape[1]//src[0].shape[1])
        if verbose > 0: print("Reduce points to: " + str(cnt))
        result = result[0,:cnt]
    else:
        raw_result = result[0]
        if verbose > 0: print("No Truncation!")
        result = result[0]

    if path != "" and verbose > 0:
        _i = 0
        vel_src = None
        '''for i in range(len(features)):
            if features[i] == 'v':
                aux_src = src[1][:,_i:_i+3]
                vel_src = aux_src/100
                _i += 3
            else:
                aux_src = src[1][:,_i:_i+1]
                _i += 1
            if verbose > 2: write_csv(path + "_%s.csv"%features[i], aux_src)'''
        
        if verbose > 0:
            permutate = True
            print(cluster_analysis(src[0][0][...,:3], raw_result, result.shape[0], permutate))

            """c = np.arange(src[0][0].shape[0])
            #c[0] = 0
            if permutate:
                c = np.repeat(c, raw_result.shape[0]//src[0][0].shape[0])[:result.shape[0]]
            else:
                c = np.repeat(np.expand_dims(c,0), raw_result.shape[0]//src[0][0].shape[0]).flatten()[:result.shape[0]]
            
            plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("detail") + ".png", src=src[0][0], vel=vel_src, z = z, c=c)"""
            fac = raw_result.shape[0]//src[0][0].shape[0]

            plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("comp") + ".png", ref=ref, src=src[0][0], vel=vel_src, z = z)
            plot_particles(result[fac:], xlim=[-1,1], ylim=[-1,1], s=5, path=path%("detail") + ".png", src=src[0][0], vel=vel_src, z = z, ref=result[:fac])
            plot_particles(src[0][0], xlim=[-1,1], ylim=[-1,1], s=5, path=path%("src") + ".png", src=src[0][0], vel=vel_src, z = z)
            if ref is not None: plot_particles(ref, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("ref") + ".png",  z = z)
            plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("res") + ".png", z = z)
            if verbose > 1:
                plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("comp") + ".svg", ref=ref, src=src[0][0], vel=vel_src, z = z)
                plot_particles(result[fac:], xlim=[-1,1], ylim=[-1,1], s=5, path=path%("detail") + ".svg", src=src[0][0], vel=vel_src, z = z, ref=result[:fac])
                plot_particles(src[0][0], xlim=[-1,1], ylim=[-1,1], s=5, path=path%("src") + ".svg", src=src[0][0], vel=vel_src, z = z)
                if ref is not None: plot_particles(ref, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("ref") + ".svg",  z = z)
                plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path%("res") + ".svg", z = z)
                if verbose > 2:
                    write_csv(path%("res") + ".csv", result)
                    if ref is not None: write_csv(path%("ref") + ".csv", ref)
                    write_csv(path%("src") + ".csv", src[0][0])

    return result

def eval_frame(model, patch_extractor, factor_d, path="", src=None, aux=None, ref=None, hdim=0, z=None, verbose=0):
    patches = patch_extractor.get_patches()
    result = model.predict(patches)
    if patch_extractor.data.shape[0] > 0:
        if z is None:
            tmp = np.repeat(patch_extractor.data, factor_d**2, axis=0)
            displace = np.transpose(np.reshape(np.mgrid[:factor_d,:factor_d,:1] + 0.5,(3,-1))) / factor_d - 0.5
            displace = np.concatenate(np.repeat(np.expand_dims(displace, axis=0), patch_extractor.data.shape[0], axis=0))
            patch_extractor.data = tmp + displace * np.array([1.,1.,0.])# + np.random.normal(scale=1/(3*factor_d), size=displace.shape)) * np.array([1.,1.,0.])
        else:
            tmp = np.repeat(patch_extractor.data, factor_d**3, axis=0)
            displace = np.transpose(np.reshape(np.mgrid[:factor_d,:factor_d,:factor_d] + 0.5,(3,-1))) / factor_d - 0.5
            displace = np.concatenate(np.repeat(np.expand_dims(displace, axis=0), patch_extractor.data.shape[0], axis=0))
            patch_extractor.data = tmp + displace# + np.random.normal(scale=1/(3*factor_d), size=displace.shape)

    if type(result) is list:
        for i in range(len(patch_extractor.positions)):
            patch_extractor.set_patch(result[0][i,:int(result[1][i] * result[0].shape[1])], i)
    else:
        for i in range(len(patch_extractor.positions)):
            cnt = int(np.count_nonzero(patches[0][...,1] != -2.0)) * (result.shape[1]//patches[0].shape[1])
            patch_extractor.set_patch(result[i,:cnt], i)
        
    result = patch_extractor.data * np.array([factor_d,factor_d, 0 if z is None else factor_d])
    if path != "" and verbose > 0:
        vel_src = None
        '''for k in aux:
            aux_src = aux[k]
            if k == 'v':
                vel_src = aux_src/100
            if verbose > 2: write_csv(path + "_%s.csv"%k, aux_src)'''

        if verbose > 0: 
            plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path%("comp") + ".png", ref=ref, src=src*factor_d, vel=vel_src, z = z)
            plot_particles(src, xlim=[0,hdim//factor_d], ylim=[0,hdim//factor_d], s=0.1, path=path%("src") + ".png", src=src, vel=vel_src, z = z)
            if ref is not None: plot_particles(ref, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path%("ref") + ".png", z = z)
            plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path%("res") + ".png", z = z)
            if verbose > 1: 
                plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path%("comp") + ".svg", ref=ref, src=src*factor_d, vel=vel_src, z = z)
                plot_particles(src, xlim=[0,hdim//factor_d], ylim=[0,hdim//factor_d], s=0.1, path=path%("src") + ".svg", src=src, vel=vel_src, z = z)
                if ref is not None: plot_particles(ref, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path%("ref") + ".svg", z = z)
                plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path%("res") + "svg", z = z)
                if verbose > 2: 
                    write_csv(path%("res") + ".csv", result)
                    if ref is not None: write_csv(path%("ref") + ".csv", ref)
                    write_csv(path%("src") + ".csv", src)
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

class NthLogger(keras.callbacks.ProgbarLogger):
    def __init__(self, batch_intervall,
                 count_mode='steps',
                 stateful_metrics=None):
        super(NthLogger, self).__init__(count_mode, stateful_metrics)
        self.batch_intervall = batch_intervall
    

    def on_train_begin(self, logs=None):
        self.verbose = True
        self.epochs = self.params['epochs']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        if self.seen % self.batch_intervall == 0 or (self.seen == 1 and self.use_steps) or (self.seen == batch_size and not self.use_steps):
            for k in self.params['metrics']:
                if k in logs:
                    self.log_values.append((k, logs[k]))

            # Skip progbar update for the last batch;
            # will be handled by on_epoch_end.
            if self.verbose and self.seen < self.target:
                self.progbar.update(self.seen, self.log_values)
            
class EvalCallback(keras.callbacks.TensorBoard):
    def __init__(self, path, src, ref, model,
                 features=[], 
                 multiple_runs=False,
                 batch_intervall=0, 
                 z=None, truncate=False,
                 verbose=0,
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
        self.eval_model = model
        self.suffix = "%s_e%03d_d%03d_t%03d" # run (opt.) - epoch - dataset - timestep
        self.path = path + "_%s" # type of output
        self.tag = os.path.basename(path) + "%s_d%03d_t%03d"
        self.src = src
        self.ref = ref
        self.features = features
        self.batch_intervall = batch_intervall
        self.z = z
        self.truncate = truncate
        self.verbose = verbose
        self.run_cnt = 0 if multiple_runs else -1
        self.run_suffix = ""

    def set_model(self, model):
        super().set_model(self.eval_model)

    def on_train_begin(self, logs={}):
        if self.run_cnt >= 0:
            self.run_suffix = "_%03d"%(self.run_cnt)       
            self.run_cnt += 1     

        for i in range(len(self.src)):
            for j in range(len(self.src[i])):
                eval_patch(self.model, self.src[i][j], self.path + self.suffix%(self.run_suffix, -1, i, j), self.ref[i][j], self.features, self.z, self.verbose, self.truncate)
    
    def on_epoch_end(self,ep,logs={}):
        super().on_epoch_end(ep, logs)
        if self.batch_intervall > 0:
            return
        print("Eval Patch")
        for i in range(len(self.src)):
            for j in range(len(self.src[i])):
                res = eval_patch(self.model, self.src[i][j], self.path + self.suffix%(self.run_suffix, ep, i, j), self.ref[i][j], self.features, self.z, self.verbose, self.truncate)
                add_images(self.writer, self.tag%(self.run_suffix, i, j), self.src[i][j][0][0], self.ref[i][j], res, ep, xlim=[-1,1], ylim=[-1,1], s=5, z=self.z)

    def on_batch_end(self,batch,logs={}):
        if self.batch_intervall <= 0 or batch % self.batch_intervall != 0 or batch > 10000:
            return
        for i in range(len(self.src)):
            for j in range(len(self.src[i])):
                res = eval_patch(self.model, self.src[i][j], self.path + self.suffix%(self.run_suffix, batch//self.batch_intervall, i, j), self.ref[i][j], self.features, self.z, self.verbose, self.truncate)
                add_images(self.writer, self.tag%(self.run_suffix, i, j), self.src[i][j][0][0], self.ref[i][j], res, batch//self.batch_intervall, xlim=[-1,1], ylim=[-1,1], s=5, z=self.z)

class EvalCompleteCallback(keras.callbacks.TensorBoard):
    def __init__(self, path, patch_extractor, ref, factor_d, hdim, model,
                 multiple_runs=False,
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
        self.eval_model = model
        #super().set_model(model)
        self.suffix = "%s_e%03d_d%03d_t%03d" # run (opt.) - epoch - dataset - timestep
        self.path = path + "_%s" # type of output
        self.tag = os.path.basename(path) + "%s_d%03d_t%03d"
        self.patch_extractor = patch_extractor
        self.ref = ref
        self.factor_d = factor_d
        self.hdim = hdim
        self.batch_intervall = batch_intervall
        self.z = z
        self.verbose = verbose
        self.run_cnt = 0 if multiple_runs else -1
        self.run_suffix = ""
    
    def set_model(self, model):
        super().set_model(self.eval_model)

    def on_train_begin(self, logs={}):
        if self.run_cnt >= 0:
            self.run_suffix = "_%03d"%(self.run_cnt)       
            self.run_cnt += 1     

        for i in range(len(self.patch_extractor)):
            for j in range(len(self.patch_extractor[i])):
                eval_frame(self.model, self.patch_extractor[i][j], self.factor_d, self.path + self.suffix%(self.run_suffix, -1, i, j), self.patch_extractor[i][j].src_data, self.patch_extractor[i][j].aux_data, self.ref[i][j], self.hdim, self.z, verbose=self.verbose)

    def on_epoch_end(self,ep,logs={}):
        super().on_epoch_end(ep, logs)
        if self.batch_intervall > 0:
            return
        print("Eval")
        for i in range(len(self.patch_extractor)):
            for j in range(len(self.patch_extractor[i])):
                res = eval_frame(self.model, self.patch_extractor[i][j], self.factor_d, self.path + self.suffix%(self.run_suffix, ep, i, j), self.patch_extractor[i][j].src_data, self.patch_extractor[i][j].aux_data, self.ref[i][j], self.hdim, self.z, verbose=self.verbose)
                add_images(self.writer, self.tag%(self.run_suffix, i, j), self.patch_extractor[i][j].src_data * self.factor_d, self.ref[i][j], res, ep, xlim=[0,self.hdim], ylim=[0,self.hdim], s=0.1, z=self.z)

    def on_batch_end(self,batch,logs={}):
        if self.batch_intervall <= 0 or batch % self.batch_intervall != 0:
            return
        for i in range(len(self.patch_extractor)):
            for j in range(len(self.patch_extractor[i])):
                res = eval_frame(self.model, self.patch_extractor[i][j], self.factor_d, self.path + self.suffix%(self.run_suffix, batch//self.batch_intervall, i, j), self.patch_extractor[i][j].src_data, self.patch_extractor[i][j].aux_data, self.ref[i][j], self.hdim, self.z, verbose=self.verbose)
                add_images(self.writer, self.tag%(self.run_suffix,i,j), self.patch_extractor[i][j].src_data * self.factor_d, self.ref[i][j], res, batch//self.batch_intervall, xlim=[0,self.hdim], ylim=[0,self.hdim], s=0.1, z=self.z)
