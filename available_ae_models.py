from unet import unet
from multiscale_unet import multiscale_unet
from v_unet import v_unet

models_dict = {
    "v_unet": v_unet,
}


def get_models_dict():
    return models_dict
