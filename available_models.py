from unet import unet
from multiscale_unet import multiscale_unet

models_dict = {
    "unet": unet,
    "multiscale_unet": multiscale_unet
}


def get_models_dict():
    return models_dict
