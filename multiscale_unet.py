import tensorflow.keras.backend as K

from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Add, Concatenate, Input, MaxPooling2D, \
    UpSampling2D
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.models import Model


def bn_relu(input, channels_last=True):
    if channels_last:
        X = BatchNormalization(axis=-1)(input)
    else:
        X = BatchNormalization(axis=1)(input)
    return Activation("relu")(X)


def multi_scale_block(input, n_filters, channels_last=True):
    axis = -1 if channels_last else 1
    norm = bn_relu(input, channels_last)

    X = Conv2D(filters=n_filters, kernel_size=1, padding="same", kernel_initializer='he_normal')(norm)
    X = Conv2D(filters=n_filters, kernel_size=1, padding="same", kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=axis)(X)

    Y = Conv2D(filters=n_filters, kernel_size=3, padding="same", kernel_initializer='he_normal')(norm)
    Y = Conv2D(filters=n_filters, kernel_size=3, padding="same", kernel_initializer='he_normal')(Y)
    Y = BatchNormalization(axis=axis)(Y)

    Z = Conv2D(filters=n_filters, kernel_size=5, padding="same", kernel_initializer='he_normal')(norm)
    Z = Conv2D(filters=n_filters, kernel_size=5, padding="same", kernel_initializer='he_normal')(Z)
    Z = BatchNormalization(axis=axis)(Z)

    concat = Concatenate(axis)([X, Y, Z])
    return Conv2D(filters=n_filters, kernel_size=3, padding="same", activation="relu", kernel_initializer='he_normal')(concat)


def residual_block(input, channels_last=True):
    X = bn_relu(input, channels_last)
    X = Conv2D(filters=512, kernel_size=3, padding="same", kernel_initializer='he_normal')(X)
    X = Conv2D(filters=512, kernel_size=3, padding="same", kernel_initializer='he_normal')(X)
    X = Add()([input, X])
    return bn_relu(X, channels_last)


def multiscale_unet(input_shape, channels_last=True):
    x_input = Input(input_shape)

    msb1 = multi_scale_block(x_input, n_filters=64, channels_last=channels_last)

    mp1 = MaxPooling2D()(msb1)
    msb2 = multi_scale_block(mp1, n_filters=128, channels_last=channels_last)

    mp2 = MaxPooling2D()(msb2)
    msb3 = multi_scale_block(mp2, n_filters=256, channels_last=channels_last)

    mp3 = MaxPooling2D()(msb3)
    msb4 = multi_scale_block(mp3, n_filters=512, channels_last=channels_last)

    rb1 = residual_block(msb4, channels_last=channels_last)
    rb2 = residual_block(rb1, channels_last=channels_last)
    rb3 = residual_block(rb2, channels_last=channels_last)

    axis = -1 if channels_last else 1

    ucr1 = Conv2D(filters=256, kernel_size=2, padding="same", activation='relu', kernel_initializer='he_normal')(UpSampling2D()(rb3))

    concat1 = Concatenate(axis)([msb3, ucr1])
    cr1 = Conv2D(filters=256, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(concat1)
    cr2 = Conv2D(filters=256, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(cr1)
    ucr2 = Conv2D(filters=128, kernel_size=2, padding="same", activation='relu', kernel_initializer='he_normal')(UpSampling2D()(cr2))

    concat2 = Concatenate(axis)([msb2, ucr2])
    cr3 = Conv2D(filters=128, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(concat2)
    cr4 = Conv2D(filters=128, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(cr3)
    ucr3 = Conv2D(filters=64, kernel_size=2, padding="same", activation='relu', kernel_initializer='he_normal')(UpSampling2D()(cr4))

    concat3 = Concatenate(axis)([msb1, ucr3])
    cr5 = Conv2D(filters=64, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(concat3)
    cr6 = Conv2D(filters=64, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(cr5)
    cr7 = Conv2D(filters=2, kernel_size=3, padding="same", activation='relu', kernel_initializer='he_normal')(cr6)
    output = Conv2D(filters=1, kernel_size=1, activation="sigmoid")(cr7)

    return Model(inputs=x_input, outputs=output)
