import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
from distutils.util import strtobool
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError

from callbacks_and_losses import custom_losses
import data

from callbacks_and_losses.custom_calllbacks import EarlyStoppingAtMinValLoss
from models.available_models import get_models_dict

models_dict = get_models_dict()


def main(args):
    start = datetime.now().strftime("%d-%m-%Y_%H.%M")
    results_dir = "results_%s" % start
    results_train_dir = os.path.join(results_dir, "results_training")
    results_train_min_loss_dir = results_train_dir + "_min_val_loss"
    results_test_dir = os.path.join(results_dir, "results_test")
    results_test_min_loss_dir = results_test_dir + "_min_val_loss"
    input_size = (None, None)
    # Load model from JSON file if file path was provided...
    if os.path.exists(args.model):
        try:
            with open(args.model, 'r') as f:
                json = f.read()
            model = model_from_json(json)
            args.model = os.path.splitext(os.path.split(args.model)[-1])[0]
        except JSONDecodeError:
            raise ValueError(
                "JSON decode error found. File path %s exists but could not be decoded; verify if JSON encoding was "
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

    # We don't resize images for training, but provide image patches of reduced size for memory saving
    # An image is turned into this size patches in a chess-board-like approach
    input_size = args.training_crop_size

    # For fine tuning, one can provide previous weights
    if args.pretrained_weights:
        model.load_weights(args.pretrained_weights)

    # Model is compiled so it can be trained
    model.compile(optimizer=Adam(lr=args.learning_rate), loss=custom_losses.bce_dsc_loss(args.alpha),
                  metrics=[custom_losses.dice_coef, 'binary_crossentropy',
                           keras_metrics.Precision(), keras_metrics.Recall()])

    # Here we find to paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    # Data is split into 80% for training data and 20% for validation data. A custom seed is used for reproducibility.
    n_training_images = int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    training_paths = paths[:, :n_training_images]
    test_paths = paths[:, n_training_images:]

    if args.save_test_paths:
        with open("test_paths.txt", "w+") as file:
            print("Test paths saved to 'test_paths.txt'")
            file.write("\n".join([";".join(paths) for paths in test_paths.transpose()]))

    # As input images can be of different sizes, here we calculate the total number of patches used for training.
    print("Calculating the total number of samples after cropping and data augmentatiton. "
          "This may take a while, don't worry.")
    n_train_samples = next(data.train_image_generator(training_paths, input_size, args.batch_size, resize=False,
                                                      count_samples_mode=True, rgb_preprocessor=None,
                                                      data_augmentation=args.use_da))
    print("\nProceeding to train.")

    # A customized early stopping callback. At each epoch end, the callback will test the current weights on the
    # validation set (using whole images instead of patches) and stop the training if the minimum validation loss hasn't
    # improved over the last 'patience' epochs.
    es = EarlyStoppingAtMinValLoss(test_paths, file_path='%s_best.hdf5' % args.model, patience=20,
                                   rgb_preprocessor=rgb_preprocessor)

    # Training begins. Note that the train image generator can use or not data augmentation through the parsed argument
    # 'use_da'
    print("Start!")
    history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size,
                                                     rgb_preprocessor=rgb_preprocessor, data_augmentation=args.use_da),
                        epochs=args.epochs,
                        verbose=1, callbacks=[es],
                        steps_per_epoch=n_train_samples // args.batch_size)
    print("Finished!")

    # Save the weights of the last training epoch. If trained on a single epoch, these weights are equal to the
    # best weights (to avoid redundancy, no new weights file is saved)
    if args.epochs > 0:
        model.save_weights("%s.hdf5" % args.model)
        print("Last epoch's weights saved.")

    # Verify if the trained model expects a particular input size
    evaluation_input_shape = tuple(model.input.shape[1:-1])
    if evaluation_input_shape == (None, None):
        evaluation_input_shape = None

    print("Evaluating the model...")
    print("On training paths:")
    data.save_results_on_paths(model, training_paths, results_train_dir)
    if args.epochs > 0:
        os.replace("%s.hdf5" % args.model, os.path.join(results_train_dir, "%s.hdf5" % args.model))
    else:
        model.save_weights(os.path.join(results_train_dir, "%s.hdf5" % args.model))
    print("\nOn test paths:")
    data.save_results_on_paths(model, test_paths, results_test_dir)
    # test_image_generator() will resize images to evaluation_input_shape if the network expects a specific input
    # size, but the default models included in this repository allow any input size
    metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, batch_size=1,
                                                         rgb_preprocessor=rgb_preprocessor),
                             steps=test_paths.shape[1])
    result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
    for idx, metric in enumerate(model.metrics_names):
        result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
    for attribute in args.__dict__.keys():
        result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(results_test_dir, "results.txt"), "w") as f:
        f.write(result_string.strip())

    # If model was trained more than one epoch, evaluate the best validation weights in the same test data. Otherwise,
    # the best validation weights are the weights at the end of the only training epoch
    if args.epochs > 1:
        # Load results using the min val loss epoch's weights
        model.load_weights('%s_best.hdf5' % args.model)
        print("Evaluating the model with minimum validation loss...")
        print("On training paths:")
        data.save_results_on_paths(model, training_paths, results_train_min_loss_dir)
        print("\nOn test paths:")
        data.save_results_on_paths(model, test_paths, results_test_min_loss_dir)
        os.replace('%s_best.hdf5' % args.model, os.path.join(results_train_min_loss_dir, '%s_best.hdf5' % args.model))
        metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, batch_size=1,
                                                             rgb_preprocessor=rgb_preprocessor),
                                 steps=test_paths.shape[1])
        result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
        for idx, metric in enumerate(model.metrics_names):
            result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
        for attribute in args.__dict__.keys():
            result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
        with open(os.path.join(results_test_min_loss_dir, "results.txt"), "w") as f:
            f.write(result_string.strip())

        # If trained more than one epoch, save the training history as csv and plot it
        print("\nPlotting training history...")
        import pandas as pd
        pd.DataFrame.from_dict(history.history).to_csv(os.path.join(results_dir, "training_history.csv"), index=False)
        # summarize history for loss
        for key in history.history.keys():
            plt.plot(history.history[key])
        plt.ylim((0.0, 1.0 + args.alpha))
        plt.title('model losses')
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.legend(history.history.keys(), loc='upper left')
        plt.savefig(os.path.join(results_dir, "training_losses.png"))
        # plt.show()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'crack500', 'gaps384', "
                             "'cracktree200'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folders containing the respective datasets as downloaded from the original "
                             "source.")
    parser.add_argument("--model", type=str, default="uvgg19",
                        help="Network to use. It can be either a name from 'models.available_models.py' or a path to a "
                             "hdf5 file.")
    parser.add_argument("--training_crop_size", type=int, nargs=2, default=[256, 256],
                        help="For memory efficiency and being able to admit multiple size images,"
                             "subimages are created by cropping original images to this size windows")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha for objective function: BCE_loss + alpha*DSC_loss")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Load previous weights from this location.")
    parser.add_argument("--use_da", type=str, default="True", help="If 'True', training will be done using data "
                                                                   "augmentation. If 'False', just raw images will be "
                                                                   "used.")
    parser.add_argument("--save_test_paths", type=str, default="False", help="If 'True', a text file 'test_paths.txt' "
                                                                             "containing the paths of the images used "
                                                                             "for testing will be saved in the "
                                                                             "project's root.")

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
