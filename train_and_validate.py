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

from callbacks_and_losses.custom_calllbacks import EarlyStoppingAtMinValLoss
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

    # model_checkpoint = ModelCheckpoint('%s_best.hdf5' % args.model, monitor='loss', verbose=1, save_best_only=True)
    # es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)

    # Here we find to paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    # Data is split into 80% for training data and 20% for validation data. A custom seed is used for reproducibility.
    n_training_images = int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    training_paths = paths[:, :n_training_images]
    test_paths = paths[:, n_training_images:]

    # As input images can be of different sizes, here we calculate the total number of patches used for training.
    n_train_samples = next(data.train_image_generator(training_paths, input_size, args.batch_size, resize=False,
                                                      count_samples_mode=True, rgb_preprocessor=None,
                                                      data_augmentation=args.use_da))

    # A customized early stopping callback. At each epoch end, the callback will test the current weights on the
    # validation set (using whole iamges instead of patches) and stop the training if the minimum validation loss hasn't
    # imporved over the last 'patience' epochs.
    es = EarlyStoppingAtMinValLoss(test_paths, file_path='%s_best.hdf5' % args.model, patience=20,
                                   rgb_preprocessor=rgb_preprocessor)

    # Training begins. Note that the train image generator can use or not data augmentation through the parsed argument
    # 'use_da'
    history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size,
                                                     rgb_preprocessor=rgb_preprocessor, data_augmentation=args.use_da),
                        epochs=args.epochs,
                        verbose=1, callbacks=[es],
                        steps_per_epoch=n_train_samples // args.batch_size)

    # Save the models of the last training epoch
    if args.epochs > 0:
        model.save_weights("%s.hdf5" % args.model)

    # Verify if the trained model expects a particular input size
    evaluation_input_shape = tuple(model.input.shape[1:-1])
    if evaluation_input_shape == (None, None):
        evaluation_input_shape = None

    # Save results using the last epoch's weights
    data.save_results_on_paths(model, training_paths, "results_training")
    data.save_results_on_paths(model, test_paths, "results_test")
    metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, batch_size=1,
                                                         rgb_preprocessor=rgb_preprocessor),
                             steps=test_paths.shape[1])
    result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
    for idx, metric in enumerate(model.metrics_names):
        result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
    for attribute in args.__dict__.keys():
        result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join("results_test", "results.txt"), "w") as f:
        f.write(result_string.strip())

    if args.epochs > 0:
        # Save results using the min val loss epoch's weights
        model.load_weights('%s_best.hdf5' % args.model)
        data.save_results_on_paths(model, training_paths, "results_training_min_val_loss")
        data.save_results_on_paths(model, test_paths, "results_test_min_val_loss")
        metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, batch_size=1,
                                                             rgb_preprocessor=rgb_preprocessor),
                                 steps=test_paths.shape[1])
        result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
        for idx, metric in enumerate(model.metrics_names):
            result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
        for attribute in args.__dict__.keys():
            result_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
        with open(os.path.join("results_test_min_val_loss", "results.txt"), "w") as f:
            f.write(result_string.strip())

    # summarize history for loss
    for key in history.history.keys():
        plt.plot(history.history[key])
    plt.ylim((0.0, 1.0 + args.alpha))
    plt.title('model losses')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(history.history.keys(), loc='upper left')
    plt.savefig("training_losses.png")
    # plt.show()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folders containing the datasets as downloaded from the original source.")
    parser.add_argument("--model", type=str, default="unet", help="Network to use.")
    parser.add_argument("--training_crop_size", type=int, nargs=2, default=[256, 256],
                        help="For memory efficiency and being able to admit multiple size images,"
                             "subimages are created by cropping original images to this size windows")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for loss BCE_loss + alpha*DSC_loss")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Load previous weights from this location.")
    parser.add_argument("--use_da", type=str, default="True", help="If 'True', training will be done using data "
                                                                   "augmentation. If 'False', just raw images will be "
                                                                   "used.")

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
