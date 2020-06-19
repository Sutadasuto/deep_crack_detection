'''
  Variational Autoencoder (VAE) with the Keras Functional API.
'''

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras import backend as K

# Define sampling with reparameterization trick
def sample_z(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps

def vae_mnist(input_size, latent_dim=2, self_supervised=True):
    img_height, img_width, num_channels = input_size
    # # =================
    # # Encoder
    # # =================

    # Definition
    i = Input(shape=(img_height, img_width, num_channels), name='encoder_input')
    cx = Conv2D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(i)
    cx = BatchNormalization()(cx)
    cx = Conv2D(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    cx = Conv2D(filters=512, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    cx = Conv2D(filters=1024, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    x = Flatten()(cx)
    x = Dense(20, activation='relu')(x)
    x = BatchNormalization()(x)
    mu = Dense(latent_dim, name='latent_mu')(x)
    sigma = Dense(latent_dim, name='latent_sigma')(x)

    # Get Conv2D shape for Conv2DTranspose operation in decoder
    conv_shape = K.int_shape(cx)

    # Use reparameterization trick to ....??
    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, sigma])

    # Instantiate encoder
    encoder = Model(i, [mu, sigma, z], name='encoder')
    # encoder.summary()

    # =================
    # Decoder
    # =================

    # Definition
    d_i = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
    x = BatchNormalization()(x)
    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx = Conv2DTranspose(filters=1024, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    cx = BatchNormalization()(cx)
    cx = Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    cx = Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    cx = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    o = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(
        cx)
    # o = Conv2D(filters=num_channels, kernel_size=5, padding='valid', name='decoder_output_adjusted')(o)

    # Instantiate decoder
    decoder = Model(d_i, o, name='decoder')
    # decoder.summary()

    # =================
    # VAE as a whole
    # =================

    # Instantiate VAE
    vae_outputs = decoder(encoder(i)[2])
    vae = Model(i, vae_outputs, name='vae')
    # vae.summary()


    # Define loss
    def kl_reconstruction_loss(true, pred):
        # Reconstruction loss
        # reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
        reconstruction_loss = mse(K.flatten(true), K.flatten(pred)) * img_width * img_height
        # KL divergence loss
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    return vae, kl_reconstruction_loss

