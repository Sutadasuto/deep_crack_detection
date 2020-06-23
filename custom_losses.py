import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy, MeanSquaredError, CosineSimilarity
from leung_malik import LeungMalik

# Smooth factor for dice coefficient. DC = (2 * GT n Pred + 1) / (GT u Pred + 1)
smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)


def bce_dsc_loss(alpha=0.5):
    def hybrid_loss(y_true, y_pred):
        dice = dice_coef_loss(y_true, y_pred)
        BCE = BinaryCrossentropy()
        bce = BCE(y_true, y_pred)
        return K.sum(bce + alpha * dice)

    return hybrid_loss


class LM_Loss(object):
    def __init__(self):
        self.leung_malik = LeungMalik()
        self.filter_bank = self.leung_malik.f
        self.lm_model = Sequential(layers=Conv2D(self.filter_bank.shape[-1],
                                         (self.filter_bank.shape[0], self.filter_bank.shape[1]),
                                         padding="same",
                                         activation="linear",
                                         use_bias=False,
                                         kernel_initializer=self.kernel_initializer,
                                         data_format="channels_last",
                                         input_shape=(None, None, 1)
                                         ))
        for layer in self.lm_model.layers:
            layer.trainable = False
        self.lm_model.build()
        self.LossFunction = MeanSquaredError()
        # self.LossFunction = CosineSimilarity()

    def kernel_initializer(self, input_shape, dtype=None):
        return np.expand_dims(self.leung_malik.make_filters().astype(np.float32), axis=2)

    def loss(self, y_true, y_pred):
        y_true = self.lm_model(y_true)
        y_pred = self.lm_model(y_pred)
        return self.LossFunction(y_true, y_pred)

# lm_loss = LM_Loss()

# def bce_kl_loss(model):
#     def kl_reconstruction_loss(y_true, y_pred):
#         # Reconstruction loss
#         reconstruction_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)) * model.input_shape[1] * model.input_shape[2]
#         # KL divergence loss
#         kl_loss = 1 + model.get_layer("latent_log_sigma").output - K.square(model.get_layer("latent_mu").output) - K.exp(model.get_layer("latent_log_sigma").output)
#         kl_loss = K.sum(kl_loss, axis=-1)
#         kl_loss *= -0.5
#         # Total loss = 50% rec + 50% KL divergence loss
#         return K.mean(reconstruction_loss + kl_loss)
#     return kl_reconstruction_loss
