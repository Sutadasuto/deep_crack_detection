import tensorflow.keras.backend as K
import tensorflow as tf

from math import log2
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *

import callbacks_and_losses.vgg_utilities as vgg


def fcae(input_shape, latent_dim=0.0625, self_supervised=True):
    if input_shape[-1] == 1:
        input_shape = (input_shape[0], input_shape[1], 3)
    n_poolings = int(log2(1/latent_dim))
    # Encoder
    inputs = Input(shape=input_shape)
    for n in range(n_poolings):
        if n == 0:
            x = Conv2D(64 * 2**n, 3, activation='relu', padding='same', name="conv%s_1" % (n+1))(inputs)
        else:
            x = Conv2D(64 * 2**n, 3, activation='relu', padding='same', name="conv%s_1" % (n+1))(x)
        x = Conv2D(64 * 2**n, 3, activation='relu', padding='same', name="conv%s_2" % (n+1))(x)
        x = AveragePooling2D(pool_size=(2, 2), name="pool%s" % (n+1))(x)
    x = Conv2D(64 * 2 ** n_poolings, 3, activation='relu', padding='same', name="conv%s_1" % (n_poolings+1))(x)
    x = Conv2D(64 * 2 ** n_poolings, 3, activation='relu', padding='same', name="conv%s_2" % (n_poolings+1))(x)
    encoder = Model(inputs, x, name="encoder")

    # Decoder
    ## Transform sampled random variables to multi-channel images
    latent = Input(shape=x.shape[1:])
    for n in range(n_poolings):
        if n == 0:
            x = Conv2D(64 * 2**((n_poolings - 1) - n), 2, activation='relu', padding='same', name="conv_d%s_0" % (n+1))\
                (UpSampling2D(size=(2, 2), name="up%s" % (n+1))(latent))
        else:
            x = Conv2D(64 * 2**((n_poolings - 1) - n), 2, activation='relu', padding='same', name="conv_d%s_0" % (n+1))\
                (UpSampling2D(size=(2, 2), name="up%s" % (n+1))(x))
        x = Conv2D(64 * 2**((n_poolings - 1) - n), 3, activation='relu', padding='same', name="conv_d%s_1" % (n+1))(x)
        x = Conv2D(64 * 2**((n_poolings - 1) - n), 3, activation='relu', padding='same', name="conv_d%s_2" % (n+1))(x)

    x = Conv2D(2*input_shape[-1], 3, activation='relu', padding='same', name="conv_d%s_3" % (n_poolings+1))(x)
    x = Conv2D(input_shape[-1], 3, activation='relu', padding='same', name="conv_d%s_4" % (n_poolings+1))(x)
    decoder = Model(latent, x, name='decoder')

    outputs = decoder(encoder(inputs))
    fcae = Model(inputs, outputs, name='fcae')

    content_layers = ['block1_conv2']
    style_layers = ['block4_conv4',
                    'block5_conv4']
                    # 'block2_conv1',
                    # 'block3_conv1',
                    # 'block4_conv1',
                    # 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    extractor = vgg.StyleContentModel(style_layers, content_layers)
    content_weight = 1e4
    style_weight = 1e-2

    def content_style_loss(y_true, y_pred):
        targets = extractor(y_true)
        preds = extractor(y_pred)
        style_loss = tf.add_n([tf.reduce_mean((preds["style"][name] - targets["style"][name]) ** 2)
                               for name in preds["style"].keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((preds["content"][name] - targets["content"][name]) ** 2)
                                 for name in preds["content"].keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    return fcae, content_style_loss
