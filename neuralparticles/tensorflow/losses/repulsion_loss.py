import tensorflow as tf
from neuralparticles.tensorflow.tools.tf_grouping import query_ball_point, group_point
from neuralparticles.tensorflow.tools.zero_mask import trunc_mask

def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss

def repulsion_loss(pred, nsample=50, radius=0.1, h=0.005):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unqiue_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)

    pts_cnt = tf.to_float(pts_cnt)
    pts_cnt = tf.expand_dims(pts_cnt, axis=-1)
    mask = trunc_mask(pts_cnt, nsample)
    dist_square = dist_square * mask + (radius ** 2) * (1-mask)

    dist_square, _ = tf.nn.top_k(-dist_square, 5)    

    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)

    #dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    
    uniform_loss = weight#radius-dist*weight
    return tf.reduce_mean(uniform_loss)

if __name__ == "__main__":
    import numpy as np
    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Flatten, Reshape, Conv1D, concatenate
    from keras.models import Model
    from neuralparticles.tools.plot_helpers import plot_particles
    from neuralparticles.tensorflow.tools.pointnet_util import pointnet_sa_module, pointnet_fp_module

    '''cnt = 101
    test = np.empty((cnt*cnt,3), dtype='float32')

    for i in range(cnt):
        for j in range(cnt):
            test[j+i*cnt] = [j,i,cnt//2]

    test = test/(cnt-1)
    test = test * 2 - 1

    np.random.shuffle(test)

    print(test)

    for i in range(1,11):
        n = np.linalg.norm(test*i,axis=-1)
        print("__")
        print(n[np.argsort(n)[1]])
        print(K.eval(repulsion_loss(np.array([test*i]), radius=0.1, h=0.05)))'''

    inputs = Input((100,3))
    activation = keras.activations.sigmoid
    fac = 4

    l1_xyz, l1_points = pointnet_sa_module(inputs, inputs, 100, 0.25, fac*4, 
                                            [fac*4,
                                            fac*4,
                                            fac*8])
    l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, 100//2, 0.5, fac*4, 
                                            [fac*8,
                                            fac*8,
                                            fac*16])
    l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, 100//4, 0.6, fac*4, 
                                            [fac*16,
                                            fac*16,
                                            fac*32])
    l4_xyz, l4_points = pointnet_sa_module(l3_xyz, l3_points, 100//8, 0.7, fac*4, 
                                            [fac*32,
                                            fac*32,
                                            fac*64])

    # interpoliere die features in l2_points auf die Punkte in x
    up_l2_points = pointnet_fp_module(inputs, l2_xyz, None, l2_points, [fac*8])
    up_l3_points = pointnet_fp_module(inputs, l3_xyz, None, l3_points, [fac*8])
    up_l4_points = pointnet_fp_module(inputs, l4_xyz, None, l4_points, [fac*8])

    x = concatenate([up_l4_points, up_l3_points, up_l2_points, l1_points, inputs], axis=-1)

    x = Conv1D(fac*32, 1, name="expansion_1", activation=activation)(x)
    x = Conv1D(fac*16, 1, name="expansion_2", activation=activation)(x)

    x = Conv1D(fac*8, 1, name="coord_reconstruction_1", activation=activation)(x)

    b = np.zeros((3,), dtype='float32')
    W = np.zeros((1, fac*8, 3), dtype='float32')
    x = Conv1D(3, 1, name="coord_reconstruction_2", activation=activation)(x)
    
    x = Reshape((100,3))(x)

    m = Model(inputs=inputs, outputs=x)
    m.compile(keras.optimizers.adam(), loss=lambda y_true, y_pred: repulsion_loss(y_pred, 20, 0.1, 0.005))

    data = np.random.random((10000,100,3))*0.1

    plot_particles(m.predict(data[:1])[0], src=data[0], s=5)
    
    m.fit(data, data, epochs=1)
    plot_particles(m.predict(data[:1])[0], src=data[0], s=5)
    m.fit(data, data, epochs=1)
    plot_particles(m.predict(data[:1])[0], src=data[0], s=5)
    m.fit(data, data, epochs=1)
    plot_particles(m.predict(data[:1])[0], src=data[0], s=5)
    m.fit(data, data, epochs=1)
    plot_particles(m.predict(data[:1])[0], src=data[0], s=5)
    m.fit(data, data, epochs=1)

    plot_particles(m.predict(data[:1])[0], src=data[0], s=5)

    