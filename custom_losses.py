import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy

# Smooth factor for dice coefficient. DC = (2 * GT n Pred + 1) / (GT u Pred )
smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)


def loss(alpha=0.5):
    def hybrid_loss(y_true, y_pred):
        dice = dice_coef_loss(y_true, y_pred)
        BCE = BinaryCrossentropy()
        bce = BCE(y_true, y_pred)
        return K.sum(bce + alpha * dice)

    return hybrid_loss