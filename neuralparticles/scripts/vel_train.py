import numpy as np 
import tensorflow as tf 
import keras 
import keras.backend as K 
from keras.layers import Input, Conv1D, concatenate
from keras.models import Model
from neuralparticles.tensorflow.losses.tf_approxmatch import approx_match, match_cost, emd_loss, approx_vel
from neuralparticles.tensorflow.losses.tf_auctionmatch import auction_match, emd_loss as emd_loss2, approx_vel as approx_vel2
import matplotlib.pyplot as plt
import os

emd = False
learn = True
inp = 100
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
def plot(data, c, r, idx):
    plt.subplot(c,r,idx)
    sx, sy = data[:,0], data[:,1]
    vx, vy = data[:,2], data[:,3]
    plt.scatter(sx, sy, s=5)
    for i in range(len(data)):
        plt.plot([sx[i],sx[i]+vx[i]],[sy[i],sy[i]+vy[i]], 'g-')

    plt.xlim([-1,1])
    plt.ylim([-1,1])


def kernel(q, h):
    c0 = tf.cast(q/h <= 1, tf.float32)
    c1 = tf.cast(tf.logical_and(q/h <= 2, q/h > 1), tf.float32)
    v = c0 * (4 - 6*q**2 + 3*q**3) + c1 * (2 - q**3)

    #return K.exp(-q/(h**2))
    return 1/(4*h**2) * v

def extract_vel(gt_pos, gt_vel, pred_pos, h=0.2):
    if not emd:
        dist = K.sum(K.square(K.expand_dims(gt_pos, axis=1) - K.expand_dims(pred_pos, axis=2)), axis=-1)
        dist = kernel(dist, h)

        w = K.clip(K.sum(dist, axis=2, keepdims=True), 1, 10000)
        return K.batch_dot(dist, gt_vel)/w
    else:
        gt_pos = concatenate([gt_pos, K.zeros_like(gt_pos)[...,:1]])
        pred_pos = concatenate([pred_pos, K.zeros_like(pred_pos)[...,:1]])
        match = approx_match(gt_pos, pred_pos)
        return K.batch_dot(match, gt_vel,[2,1])

def vel_loss(y_true, y_pred):
    """gt_pos = concatenate([y_true[...,:2], K.zeros_like(y_true[...,:2])[...,:1]])
    pred_pos = concatenate([y_pred[...,:2], K.zeros_like(y_pred[...,:2])[...,:1]])
    match = approx_match(gt_pos, pred_pos)
    return match_cost(concatenate([y_true[...,2:], K.zeros_like(y_true[...,:2])[...,:1]]), concatenate([y_pred[...,2:], K.zeros_like(y_pred[...,:2])[...,:1]]), match)/ tf.cast(tf.shape(y_pred)[1], tf.float32)"""
    pred_pos, pred_vel = y_pred[...,:2], y_pred[...,2:]
    true_pos, true_vel = y_true[...,:2], y_true[...,2:]
    return K.mean(K.abs(extract_vel(true_pos, true_vel, pred_pos)-pred_vel), axis=-1)

inputs = Input((inp,2))

x = Conv1D(8, 1, activation="relu")(inputs)
x = Conv1D(16, 1, activation="relu")(x)
x = Conv1D(32, 1, activation="relu")(x)
x = Conv1D(16, 1, activation="relu")(x)
x = Conv1D(8, 1, activation="relu")(x)
x = Conv1D(2, 1, activation="tanh")(x)

outputs = concatenate([inputs,x])

m = Model(inputs, outputs)

m.compile(keras.optimizers.adam(lr=0.001), loss=vel_loss)
m.summary()

np.random.seed(100)

gt = (np.random.random((10000,10,4)) - 0.5) * 2
gt[...,2:] = gt[...,:2]
src = (np.random.random((10000,inp,2)) - 0.5) * 2

test = (np.random.random((10,inp,4)) - 0.5) * 2
test[...,2:] = test[...,:2]


a = K.constant(np.concatenate([src[:10,:,:2], np.zeros_like(src[:10,:,:2])[...,:1]], axis=-1))
b = K.constant(np.concatenate([test[:10,:,:2], np.zeros_like(test[:10,:,:2])[...,:1]], axis=-1))

print("_---")
print(K.eval(K.mean(emd_loss(a,b))))
print(K.eval(K.mean(emd_loss2(a,b))))
print(K.eval(K.mean(emd_loss2(b,a))))

"""print(K.eval(K.mean(emd_loss(b,a))))
print(K.eval(K.mean(emd_loss(a[:,:10],b))))
print(K.eval(K.mean(emd_loss(a,b[:,:10]))))
print(K.eval(K.mean(emd_loss(a[:,:10],b[:,:10]))))
print(K.eval(K.mean(emd_loss(a,b, tf.tile(K.constant([10],dtype=tf.int32), tf.shape(a)[:1]), None))))
print(K.eval(K.mean(emd_loss(a,b, None, tf.tile(K.constant([10],dtype=tf.int32), tf.shape(a)[:1])))))
print(K.eval(K.mean(emd_loss(a,b, tf.tile(K.constant([10],dtype=tf.int32), tf.shape(a)[:1]), tf.tile(K.constant([10],dtype=tf.int32), tf.shape(a)[:1])))))"""

v = K.constant(1)#K.constant(np.random.random((10,100,3))-0.5)*0.1
print(K.eval(K.mean(approx_vel(a,a+v))))
print(K.eval(K.mean(v)))
print(K.eval(K.max(K.abs(approx_vel(a,a+v)-v))))

print(K.eval(K.max(K.abs(approx_vel2(a,a+v)-v))))

bl = (np.random.random((10,4)) - 0.5) * 2
bl[:] = gt[0]
np.random.shuffle(bl)
bl = np.expand_dims(bl, axis=0)

gt_pos = K.constant(np.concatenate([gt[:1,:,:2], np.zeros_like(gt[:1,:,:2])[...,:1]], axis=-1))
pred_pos = K.constant(np.concatenate([bl[:,:,:2], np.zeros_like(bl[:,:,:2])[...,:1]], axis=-1))
#print(K.eval(approx_match(gt_pos, pred_pos)))
input2 = Input((10,3))

x = Conv1D(8, 1, activation="relu")(input2)
x = Conv1D(16, 1, activation="relu")(x)
x = Conv1D(32, 1, activation="relu")(x)
x = Conv1D(16, 1, activation="relu")(x)
x = Conv1D(8, 1, activation="relu")(x)
x = Conv1D(3, 1, activation="tanh")(x)

m2 = Model(input2, x)

m2.compile(keras.optimizers.adam(lr=0.001), loss=emd_loss)

gt = np.concatenate((gt[...,:2], np.zeros_like(gt[...,:1])), axis=-1)
m2.fit(gt, gt, batch_size=32, epochs=10)

print(K.eval(K.mean(emd_loss(gt_pos, pred_pos))))
print(K.eval(K.mean(emd_loss(pred_pos, gt_pos))))
print(m2.evaluate(x=np.concatenate([gt[:1,:,:2], np.zeros_like(gt[:1,:,:2])[...,:1]], axis=-1), y=np.concatenate([bl[:,:,:2], np.zeros_like(bl[:,:,:2])[...,:1]], axis=-1)))
#print(K.eval(K.mean(emd_loss(gt_pos, pred_pos, tf.tile(K.constant([10],dtype=tf.int32), tf.shape(gt_pos)[:1]), tf.tile(K.constant([10],dtype=tf.int32), tf.shape(pred_pos)[:1])))))
exit()
if not learn:
    predicted = np.empty_like(test)
    predicted[:] = test[:]
    predicted[...,2:] = K.eval(extract_vel(K.constant(gt[:10,:,:2]), K.constant(gt[:10,:,2:]), K.constant(test[...,:2])))
else:
    m.fit(x=src, y=gt, batch_size=32, epochs=10)
    predicted = m.predict(test[...,:2])

print(np.mean(np.linalg.norm(test[:,2:] - predicted[:,2:], axis=-1)))
for i in range(5):
    plot(test[i], 5, 3, 1 + i*3)
    plot(predicted[i], 5, 3, 2 + i*3)
    plot(gt[i], 5, 3, 3 + i*3)
plt.show()






