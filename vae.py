import tensorflow.keras.backend as K

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *

from tensorflow.keras.optimizers import Adam


# Define sampling with reparameterization trick
def sample_z(args):
    mu, log_sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
    return mu + K.exp(log_sigma / 2) * eps


def vae(input_shape, latent_dim=2, self_supervised=True):
    # Encoder
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    ## Get latent space random variables
    conv_shape = K.int_shape(conv5)
    flatten = Flatten()(drop5)
    # x = Dense(latent_dim*10, activation='relu')(flatten)
    # mu = Dense(latent_dim, name='latent_mu', activation='linear')(x)
    mu = Dense(latent_dim, name='latent_mu', activation='linear')(flatten)
    # log_sigma = Dense(latent_dim, name='latent_log_sigma', activation='linear')(x)
    log_sigma = Dense(latent_dim, name='latent_log_sigma', activation='linear')(flatten)
    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, log_sigma])
    encoder = Model(inputs, [z, mu, log_sigma], name="encoder")

    # Decoder
    ## Transform sampled random variables to multi-channel images
    latent = Input(shape=z.shape[-1])
    y = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(latent)
    unflatten = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(y)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(unflatten))
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    decoder = Model([inputs, latent], conv10, name='decoder')

    outputs = decoder([inputs, encoder(inputs)[0]])
    vae = Model(inputs, outputs, name='vae')

    def kl_loss():
        loss = 1 + log_sigma - K.square(mu) - K.exp(log_sigma)
        loss = K.sum(loss, axis=-1)
        loss *= -0.5
        return K.mean(loss)

    if self_supervised:
        # def reconstruction_loss(y, y_decoded):
        #     loss = mse(K.flatten(y), K.flatten(y_decoded))
        #     # loss *= inputs.shape[1] * inputs.shape[2]
        #     return loss
        from custom_losses import LM_Loss
        reconstruction_loss = LM_Loss().loss

    else:
        from custom_losses import dice_coef_loss
        def reconstruction_loss(y, y_decoded):
            alpha = 0.5
            loss = binary_crossentropy(K.flatten(y), K.flatten(y_decoded))
            loss += alpha * dice_coef_loss(y, y_decoded)
            loss /= (1 + alpha)
            # loss *= inputs.shape[1] * inputs.shape[2]
            return loss

    def my_loss(y, y_decoded):
        beta = 1
        return beta * reconstruction_loss(y, y_decoded) + (1 - beta) * kl_loss()

    return vae, my_loss
