from vae import vae
from v_unet import v_unet

models_dict = {
    "v_unet": v_unet,
    "vae": vae
}


def get_models_dict():
    return models_dict
