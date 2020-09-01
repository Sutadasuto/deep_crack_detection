import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os

from distutils.util import strtobool
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError

from callbacks_and_losses import custom_losses
import data

from models.available_models import get_models_dict

models_dict = get_models_dict()


def main(args):
    input_size = (None, None)
    # Load model from JSON file if file path was provided...
    if os.path.exists(args.model):
        try:
            with open(args.model, 'r') as f:
                json = f.read()
            model = model_from_json(json)
        except JSONDecodeError:
            raise ValueError("JSON decode error found. File path %s exists but could not be decoded; verify if JSON encoding was "
                  "performed properly." % args.model)
    # ...Otherwise, create model from this project by using a proper key name
    else:
        model = models_dict[args.model]((input_size[0], input_size[1], 1))
    try:
        # Model name should match with the name of a model from
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/
        # This assumes you used a model with RGB inputs as the first part of your model,
        # therefore your input data should be preprocessed with the corresponding
        # 'preprocess_input' function
        m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
        rgb_preprocessor = getattr(m, "preprocess_input")
    except ModuleNotFoundError:
        rgb_preprocessor = None

    # Load trained weights
    model.load_weights(args.pretrained_weights)

    # Model is compiled to provide the desired metrics
    model.compile(optimizer=Adam(lr=1e-4), loss=custom_losses.bce_dsc_loss(3.0),
                  metrics=[custom_losses.dice_coef, keras_metrics.Precision(), keras_metrics.Recall()])

    # Here we find to paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)

    # Verify if the trained model expects a particular input size
    evaluation_input_shape = tuple(model.input.shape[1:-1])
    if evaluation_input_shape == (None, None):
        evaluation_input_shape = None

    print("Evaluating the model...")
    data.save_results_on_paths(model, paths, args.save_to)
    metrics = model.evaluate(x=data.test_image_generator(paths, evaluation_input_shape, batch_size=1,
                                                         rgb_preprocessor=rgb_preprocessor),
                             steps=paths.shape[1])
    result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
    for idx, metric in enumerate(model.metrics_names):
        result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
    for attribute in args.__dict__.keys():
        result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(args.save_to, "results.txt"), "w") as f:
        f.write(result_string.strip())


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("dataset_paths", type=str, nargs="+",
                        help="Path to the folders containing the datasets as downloaded from the original source.")
    parser.add_argument("model", type=str, help="Network to use.")
    parser.add_argument("pretrained_weights", type=str,
                        help="Load trained weights from this location.")
    parser.add_argument("--save_to", type=str, default="results",
                        help="Save results in this location (folder is created if it doesn't exist).")

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
