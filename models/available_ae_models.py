from models.vae import vae
from models.v_unet import v_unet
from models.vae_mnist import vae_mnist
from models.fcae import fcae

models_dict = {
    "v_unet": v_unet,
    "vae": vae,
    "vae_mnist": vae_mnist,
    "fcae": fcae
}


def get_models_dict():
    return models_dict
