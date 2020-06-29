import argparse
import cv2
import numpy as np
import os

from distutils.util import strtobool
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError

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
            raise ValueError(
                "JSON decode error found. File path %s exists but could not be decoded; verify if JSON encoding was "
                "performed properly." % args.model)
    # ...Otherwise, create model from this project by using a proper key name
    else:
        model = models_dict[args.model]((input_size[0], input_size[1], 1))

    model.load_weights(args.weights_path)

    [im, gt, pred] = data.test_image_from_path(model, args.image_path, args.gt_path)

    if gt is not None:
        result = 255 * np.concatenate([im, gt, pred], axis=1)
    else:
        result = 255 * np.concatenate([im, pred], axis=1)

    if args.save_to:
        if args.save_to == "result_imName":
            args.save_to = "result_%s" % os.path.split(args.image_path)[-1]
        path, filename = os.path.split(args.save_to)
        if not os.path.exists(path) and path != '':
            os.makedirs(path)
        cv2.imwrite(args.save_to, result)

    if args.show_result:
        import matplotlib.pyplot as plt
        imgplot = plt.imshow(result, cmap="gray")
        plt.show()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image to evaluate.")
    parser.add_argument("model", type=str, help="Network to use.")
    parser.add_argument("weights_path", type=str, help="Path to pre-trained weights to use.")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Path to the Ground Truth if the user wants to use it for comparison. None otherwise.")
    parser.add_argument("--show_result", type=str, default="False",
                        help="'True' or 'False'. If True, the input image, GT (if provided) and the prediction will be "
                             "shown together in a new screen.")
    parser.add_argument("--save_to", type=str, default="result_imName", help="Save the comparison image to this location. "
                                                                          "If 'None', no image will be saved")
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
