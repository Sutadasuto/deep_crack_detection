from models.unet import unet
from models.multiscale_unet import multiscale_unet
from models.uvgg19 import uvgg19

models_dict = {
    "unet": unet,
    "multiscale_unet": multiscale_unet,
    "uvgg19": uvgg19
}


def get_models_dict():
    return models_dict
