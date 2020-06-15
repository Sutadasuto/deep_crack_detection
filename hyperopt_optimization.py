import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from distutils.util import strtobool
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as keras_metrics

import custom_losses
import data

from custom_calllbacks import EarlyStoppingAtMinValLoss
from available_ae_models import get_models_dict

import tensorflow.keras.backend as K

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

models_dict = get_models_dict()


def main(args):
    input_size = tuple(args.resize_inputs_to)

    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    if args.self_supervised:
        paths[1, :] = paths[0, :]
    n_training_images = int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    training_paths = paths[:, :n_training_images]
    test_paths = paths[:, n_training_images:]

    n_train_samples = next(data.train_image_generator(training_paths, input_size, params["batch_size"], resize=True,
                                                      count_samples_mode=True))

    space = {'alpha': hp.choice('alpha', [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]),
             'activation': hp.choice('activation', ['relu', 'sigmoid', 'linear']),
             'batch_size': hp.choice('batch_size', [1, 2, 4])}

    def f_nn(params):

        input_shape = (256, 256, 1)
        inputs = Input(shape=(input_size[0], input_size[1], 1))
        conv1 = Conv2D(64, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation=params["activation"], padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation=params["activation"], padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation=params["activation"], padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation=params["activation"], padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation=params["activation"], padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10, name="u-net")

        my_loss = custom_losses.bce_dsc_loss(params["alpha"])

        metrics_list = [custom_losses.dice_coef, keras_metrics.Precision(), keras_metrics.Recall()]
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=my_loss, metrics=metrics_list
                      , experimental_run_tf_function=False
                      )

        es = EarlyStoppingAtMinValLoss(test_paths, file_path=None, patience=20)
        model.fit(x=data.train_image_generator(training_paths, input_size, params["batch_size"]), epochs=args.epochs,
                  verbose=0,
                  callbacks=[es],
                  steps_per_epoch=n_train_samples // params["batch_size"])

        evaluation_input_shape = tuple(model.input.shape[1:-1])
        if evaluation_input_shape == (None, None):
            evaluation_input_shape = None
        metrics = model.evaluate(x=data.test_image_generator(test_paths, evaluation_input_shape, 1),
                                 steps=test_paths.shape[1], verbose=0)

        for idx, metric in enumerate(model.metrics_names):
            if metric == "dice_coef":
                loss = metrics[idx]
                break

        tf.keras.backend.clear_session()
        print('Test DSC:', loss)
        return {'loss': loss, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=args.fmin_max_evals, trials=trials)
    print('best: ')
    print(space_eval(space, best))

    result = "Best loss: %s\nParameters: %s\n\nLoss, Parameters\n" % (trials.best_trial["result"]["loss"], space_eval(space, best))
    for trial in range(len(trials)):
        trial_result = trials.results[trial]['loss']
        trial_dict = {}
        for key in trials.vals.keys():
            trial_dict[key] = trials.vals[key][trial]
        result += "%s, %s\n" % (trial_result, space_eval(space, trial_dict))
    with open("hyperparameter_search_result.txt", "w") as f:
        f.write(result.strip())


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folders containing the datasets as downloaded from the original source.")
    parser.add_argument("--fmin_max_evals", type=int, default=10, help="'max_evals' argument value for hyperopt's "
                                                                       "fmin function.")
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
