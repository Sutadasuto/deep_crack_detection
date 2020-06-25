from unet import unet
from multiscale_unet import multiscale_unet
from uvgg19 import uvgg19

models_dict = {
    "unet": unet,
    "multiscale_unet": multiscale_unet,
    "uvgg19": uvgg19
}


def get_models_dict():
    return models_dict
