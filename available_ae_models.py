from vae import vae
from v_unet import v_unet
from vae_mnist import vae_mnist

models_dict = {
    "v_unet": v_unet,
    "vae": vae,
    "vae_mnist": vae_mnist
}


def get_models_dict():
    return models_dict
