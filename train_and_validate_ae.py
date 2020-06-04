import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from distutils.util import strtobool
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import custom_losses
import data

from custom_calllbacks import EarlyStoppingAtMinValLoss
from available_ae_models import get_models_dict

models_dict = get_models_dict()


def main(args):
    input_size = tuple(args.resize_inputs_to)
    model, loss = models_dict[args.model]((input_size[0], input_size[1], 1))
    if args.pretrained_weights:
        model.load_weights(args.pretrained_weights)
    model.compile(optimizer=Adam(lr=args.learning_rate), loss=loss, metrics=['mse']
                    , experimental_run_tf_function = False
                  )

    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    if args.self_supervised:
        paths[1, :] = paths[0, :]
    n_training_images = int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    training_paths = paths[:, :n_training_images]
    test_paths = paths[:, n_training_images:]

    n_train_samples = next(data.train_image_generator(training_paths, input_size, args.batch_size, resize=True,
                                                      count_samples_mode=True))
    es = EarlyStoppingAtMinValLoss(test_paths, file_path='%s_best.hdf5' % args.model, patience=20)

    history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size), epochs=args.epochs,
                        verbose=2,
                        callbacks=[es],
                        steps_per_epoch=n_train_samples // args.batch_size)
    model.save_weights("%s.hdf5" % args.model)

    evaluation_input_shape = tuple(model.input.shape[1:-1])
    if evaluation_input_shape == (None, None):
        evaluation_input_shape = None

    # Save results using the last epoch's weights
    data.save_results_on_paths(model, training_paths, "results_training")
    data.save_results_on_paths(model, test_paths, "results_test")
    metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, 1),
                             steps=test_paths.shape[1])
    result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
    for idx, metric in enumerate(model.metrics_names):
        result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
    with open(os.path.join("results_test", "results.txt"), "w") as f:
        f.write(result_string.strip())

    # Save results using the min val loss epoch's weights
    model.load_weights('%s_best.hdf5' % args.model)
    data.save_results_on_paths(model, training_paths, "results_training_min_val_loss")
    data.save_results_on_paths(model, test_paths, "results_test_min_val_loss")
    metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, 1),
                             steps=test_paths.shape[1])
    result_string = "Dataset: %s\nModel: %s\n" % ("/".join(args.dataset_names), args.model)
    for idx, metric in enumerate(model.metrics_names):
        result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
    with open(os.path.join("results_test_min_val_loss", "results.txt"), "w") as f:
        f.write(result_string.strip())

    # summarize history for loss
    for key in history.history.keys():
        plt.plot(history.history[key])
    # plt.ylim((0.0, 1.0 + args.alpha))
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
    parser.add_argument("--model", type=str, default="v_unet", help="Network to use.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Load previous weights from this location.")
    parser.add_argument("--resize_inputs_to", type=int, nargs=2, default=[256, 256], help="Resize images to this "
                                                                                           "dimensions (commonly used "
                                                                                           "when the network requires a "
                                                                                           "specific input size)")
    parser.add_argument("--self_supervised", type=str, default="True",
                        help="If 'True', the input images will be used as training target instead of GT annotations.")

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

