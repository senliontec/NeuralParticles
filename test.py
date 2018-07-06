import tensorflow as tf
import keras
import io
import matplotlib.pyplot as plt
from neuralparticles.tools.plot_helpers import plot_particles
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K
from keras.callbacks import TensorBoard

def make_image(points):
    """Create a pyplot plot and save to buffer."""
    imgs = []
    buf = io.BytesIO()
    plot_particles(points, path=buf)
    image_string = buf.getvalue()
    imgs.append(tf.Summary.Image(colorspace=4, encoded_image_string=image_string))
    plot_particles(points, path=buf)
    image_string = buf.getvalue()
    imgs.append(tf.Summary.Image(colorspace=4, encoded_image_string=image_string))
    buf.close()
    return imgs

class TensorBoardImage(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super().__init__(log_dir,
                         histogram_freq,
                         batch_size,
                         write_graph,
                         write_grads,
                         write_images,
                         embeddings_freq,
                         embeddings_layer_names,
                         embeddings_metadata)
                

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        image = make_image(np.random.random((100,2)))
        summary = tf.Summary(value=[tf.Summary.Value(tag="blub", image=image[0]),tf.Summary.Value(tag="blub2", image=image[1])])
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

        return

tbi_callback = TensorBoardImage('./logs',write_grads=True)

inputs = Input((10,))
x = Dense(10)(inputs)
m = Model(inputs=inputs,outputs=x)
m.compile(optimizer=keras.optimizers.adam(lr=0.1), loss='mse')
m.fit(np.random.random((10,10)), np.random.random((10,10)), epochs=4, callbacks=[tbi_callback])