import keras

from neuralparticles.tools.plot_helpers import plot_particles, write_csv
from neuralparticles.tools.data_helpers import PatchExtractor

def eval_patch(model, src, path="", ref=None, features=[]):
    #print(src)
    result = model.predict(src)
    #print(result)
    if path != "":
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
            write_csv(path + "_%s.csv"%features[i], aux_src)
        
        if type(result) is list:
            print("Truncate points at %d" % int(result[1][0] * result[0].shape[1]))
            result = result[0][0,:int(result[1][0] * result[0].shape[1])]
        else:
            result = result[0]

        plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path + ".pdf", ref=ref, src=src[0][0], vel=vel_src)
        plot_particles(result, xlim=[-1,1], ylim=[-1,1], s=5, path=path + ".png", ref=ref, src=src[0][0], vel=vel_src)
        write_csv(path + "_res.csv", result)
        write_csv(path + "_ref.csv", ref)
        write_csv(path + "_src.csv", src[0][0])

    return result

def eval_frame(model, patch_extractor, factor_d, path="", src=None, aux=None, ref=None, hdim=0):
    while(True):
        s = patch_extractor.get_patch()
        if s is None:
            break
        result = model.predict(x=s)
        if type(result) is list:
            result = result[0][0,:int(result[1][0] * result[0].shape[1])]
        else:
            result = result[0]
        patch_extractor.set_patch(result)
    result = patch_extractor.data * factor_d
    if path != "":
        vel_src = None
        for k in aux:
            aux_src = aux[k]
            if k == 'v':
                vel_src = aux_src/100
            write_csv(path + "_%s.csv"%k, aux_src)

        plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path + ".pdf", ref=ref, src=src*factor_d, vel=vel_src)
        plot_particles(result, xlim=[0,hdim], ylim=[0,hdim], s=0.1, path=path + ".png", ref=ref, src=src*factor_d, vel=vel_src)
        write_csv(path + "_res.csv", result)
        write_csv(path + "_ref.csv", ref)
        write_csv(path + "_src.csv", src)
    patch_extractor.reset()
    return result

class NthLogger(keras.callbacks.Callback):
    def __init__(self,model,li=10,cpi=100,cpt_path="model", offset=0):
        self.act = offset
        self.li = li
        self.cpi = cpi
        self.cpt_path = cpt_path
        self.model = model

    def on_epoch_end(self,batch,logs={}):
        self.act += 1
        if self.act % self.li == 0 or self.act == 1:
            print('%d/%d - loss: %f val_loss: %f' % (self.act, self.params['epochs'], logs['loss'], logs['val_loss']))
        if self.act % self.cpi == 0:
            path = "%s_%04d.h5" % (self.cpt_path, self.act//self.cpi)
            self.model.save(path)
            print('Saved Checkpoint: %s' % path)

class EvalCallback(keras.callbacks.Callback):
    def __init__(self, path, model, src, ref, features=[], batch_intervall=0):
        self.path = path
        self.model = model
        self.src = src
        self.ref = ref
        self.features = features
        self.batch_intervall = batch_intervall
        for i in range(len(self.src)):
            eval_patch(self.model, self.src[i], self.path%(i,0), self.ref[i], self.features)
    
    def on_epoch_end(self,ep,logs={}):
        if self.batch_intervall > 0:
            return
        print("Eval Patch")
        for i in range(len(self.src)):
            eval_patch(self.model, self.src[i], self.path%(i,ep+1), self.ref[i], self.features)

    def on_batch_end(self,batch,logs={}):
        if self.batch_intervall <= 0 or batch % self.batch_intervall != 0 or batch > 10000:
            return
        for i in range(len(self.src)):
            eval_patch(self.model, self.src[i], self.path%(i,(batch//self.batch_intervall)+1), self.ref[i], self.features)

class EvalCompleteCallback(keras.callbacks.Callback):
    def __init__(self, path, model, patch_extractor, ref, factor_d, hdim, batch_intervall=0):
        self.path = path
        self.model = model
        self.patch_extractor = patch_extractor
        self.ref = ref
        self.factor_d = factor_d
        self.hdim = hdim
        self.batch_intervall = batch_intervall
        for i in range(len(self.patch_extractor)):
            eval_frame(self.model, self.patch_extractor[i], self.factor_d, self.path%(i,0), self.patch_extractor[i].src_data, self.patch_extractor[i].aux_data, self.ref[i], self.hdim)
    
    def on_epoch_end(self,ep,logs={}):
        if self.batch_intervall > 0:
            return
        print("Eval")
        for i in range(len(self.patch_extractor)):
            eval_frame(self.model, self.patch_extractor[i], self.factor_d, self.path%(i,ep+1), self.patch_extractor[i].src_data, self.patch_extractor[i].aux_data, self.ref[i], self.hdim)

    def on_batch_end(self,batch,logs={}):
        if self.batch_intervall <= 0 or batch % self.batch_intervall != 0:
            return
        for i in range(len(self.patch_extractor)):
            eval_frame(self.model, self.patch_extractor[i], self.factor_d, self.path%(i,(batch//self.batch_intervall)+1), self.patch_extractor[i].src_data, self.patch_extractor[i].aux_data, self.ref[i], self.hdim)