import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from distutils.util import strtobool
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
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
iteration = 0


def main(args):
    input_size = (256, 256)

    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    n_training_images = int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    training_paths = paths[:, :n_training_images]
    test_paths = paths[:, n_training_images:]

    n_train_samples = next(data.train_image_generator(training_paths, input_size, 1, resize=False,
                                                      count_samples_mode=True))

    from itertools import chain, combinations

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    space = {'alpha': hp.choice('alpha', [0.3, 3.0]),
             'learning_rate': hp.choice('learning_rate', [1e-3, 1e-4]),
             'optimizer': hp.choice('optimizer', [SGD, RMSprop, Adam, Adadelta, Adagrad]),
             'batch_normalization_layers': hp.choice('batch_normalization_layers', list(powerset([i for i in range(1, 6)]))),
             'dropout_layers': hp.choice('dropout_layers', list(powerset([i for i in range(1, 6)]))),
             'dropout': hp.uniform('dropout', 0.0, 1.0),
             'pooling': hp.choice('pooling', [True, False])}

    def f_nn(params):

        strides = 1 if params["pooling"] else 2

        inputs = Input(shape=(None, None, 1))
        conv1 = Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
        conv1_1 = Conv2D(64, 3, activation="relu", padding='same', strides=strides, kernel_initializer='he_normal')(conv1)
        if params["pooling"]:
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
        else:
            pool1 = conv1_1
        if 1 in params["batch_normalization_layers"]:
            pool1 = BatchNormalization()(pool1)
        if 1 in params["dropout_layers"]:
            pool1 = Dropout(params["dropout"])(pool1)
        conv2 = Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool1)
        conv2_1 = Conv2D(128, 3, activation="relu", padding='same', strides=strides, kernel_initializer='he_normal')(conv2)
        if params["pooling"]:
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
        else:
            pool2 = conv2_1
        if 2 in params["batch_normalization_layers"]:
            pool2 = BatchNormalization()(pool2)
        if 2 in params["dropout_layers"]:
            pool2 = Dropout(params["dropout"])(pool2)
        conv3 = Conv2D(256, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool2)
        conv3_1 = Conv2D(256, 3, activation="relu", padding='same', strides=strides, kernel_initializer='he_normal')(conv3)
        if params["pooling"]:
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)
        else:
            pool3 = conv3_1
        if 3 in params["batch_normalization_layers"]:
            pool3 = BatchNormalization()(pool3)
        if 3 in params["dropout_layers"]:
            pool3 = Dropout(params["dropout"])(pool3)
        conv4 = Conv2D(512, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool3)
        conv4_1 = Conv2D(512, 3, activation="relu", padding='same', strides=strides, kernel_initializer='he_normal')(conv4)
        if params["pooling"]:
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
        else:
            pool4 = conv4_1
        if 4 in params["batch_normalization_layers"]:
            pool4 = BatchNormalization()(pool4)
        if 4 in params["dropout_layers"]:
            pool4 = Dropout(params["dropout"])(pool4)

        conv5 = Conv2D(1024, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv5)
        if 5 in params["batch_normalization_layers"]:
            conv5 = BatchNormalization()(conv5)
        if 5 in params["dropout_layers"]:
            conv5 = Dropout(params["dropout"])(conv5)

        up6 = Conv2D(512, 2, activation="relu", padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv5))
        if 5 in params["batch_normalization_layers"]:
            up6 = BatchNormalization()(up6)
        if 5 in params["dropout_layers"]:
            up6 = Dropout(params["dropout"])(up6)
        if params["pooling"]:
            merge6 = concatenate([conv4_1, up6], axis=3)
        else:
            merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation="relu", padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        if 4 in params["batch_normalization_layers"]:
            up7 = BatchNormalization()(up7)
        if 4 in params["dropout_layers"]:
            up7 = Dropout(params["dropout"])(up7)
        if params["pooling"]:
            merge7 = concatenate([conv3_1, up7], axis=3)
        else:
            merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation="relu", padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        if 3 in params["batch_normalization_layers"]:
            up8 = BatchNormalization()(up8)
        if 3 in params["dropout_layers"]:
            up8 = Dropout(params["dropout"])(up8)
        if params["pooling"]:
            merge8 = concatenate([conv2_1, up8], axis=3)
        else:
            merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation="relu", padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        if 2 in params["batch_normalization_layers"]:
            up9 = BatchNormalization()(up9)
        if 2 in params["dropout_layers"]:
            up9 = Dropout(params["dropout"])(up9)
        if params["pooling"]:
            merge9 = concatenate([conv1_1, up9], axis=3)
        else:
            merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
        if 1 in params["batch_normalization_layers"]:
            conv9 = BatchNormalization()(conv9)
        if 1 in params["dropout_layers"]:
            conv9 = Dropout(params["dropout"])(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10, name="u-net")

        my_loss = custom_losses.bce_dsc_loss(params["alpha"])

        metrics_list = [custom_losses.dice_coef, keras_metrics.Precision(), keras_metrics.Recall()]
        model.compile(optimizer=params["optimizer"](lr=params["learning_rate"]), loss=my_loss, metrics=metrics_list
                      , experimental_run_tf_function=False
                      )

        es = EarlyStoppingAtMinValLoss(test_paths, file_path=None, patience=20)
        history = model.fit(x=data.train_image_generator(training_paths, input_size, 4, resize=False),
                  epochs=args.epochs,
                  verbose=0,
                  callbacks=[es],
                  steps_per_epoch=n_train_samples // 4)

        metrics = model.evaluate(x=data.test_image_generator(test_paths, input_size=None, batch_size=1),
                                 steps=test_paths.shape[1], verbose=0)

        for idx, metric in enumerate(model.metrics_names):
            if metric == "dice_coef":
                loss = metrics[idx]
                break

        try:
            with open("best_dsc.txt", "r") as f:
                best_dsc = float(f.read())
        except FileNotFoundError:
            best_dsc = 0

        if loss > best_dsc:
            with open("best_dsc.txt", "w") as f:
                f.write(str(loss))
            model_json = model.to_json()
            with open("best_dsc_model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("best_dsc_weights.hdf5")
            data.save_results_on_paths(model, training_paths, "results_training")
            data.save_results_on_paths(model, test_paths, "results_test")

        plt.clf()
        for key in history.history.keys():
            if key in ["val_dice_coef", "dice_coef"]:
                plt.plot(history.history[key])
            # plt.ylim((0.0, 1.0 + args.alpha))
        plt.ylim((0.0, 1.0))
        plt.title('model losses')
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.legend([key for key in history.history.keys() if key in ["val_dice_coef", "dice_coef"]], loc='upper left')
        global iteration
        iteration += 1
        if not os.path.exists("training_histories"):
            os.makedirs("training_histories")
        plt.savefig(os.path.join("training_histories", "history_iteration_%s.png" % iteration))

        tf.keras.backend.clear_session()
        print('Test DSC:', loss)
        print('Params: %s' % str(params))
        return {'loss': -loss, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=args.fmin_max_evals, trials=trials)
    print('best: ')
    print(space_eval(space, best))

    result = "Best DSC: {:.4f}\nParameters: {}\n\nDSC, Parameters\n".format(-trials.best_trial["result"]["loss"], space_eval(space, best))
    for trial in range(len(trials)):
        trial_result = trials.results[trial]['loss']
        trial_dict = {}
        for key in trials.vals.keys():
            trial_dict[key] = trials.vals[key][trial]
        result += "{:.4f}, {}\n".format(-trial_result, space_eval(space, trial_dict))
    with open("hyperparameter_search_result.txt", "w") as f:
        f.write(result.strip())

    if os.path.exists("best_dsc.txt"):
        os.remove("best_dsc.txt")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folders containing the datasets as downloaded from the original source.")
    parser.add_argument("--fmin_max_evals", type=int, default=10, help="'max_evals' argument value for hyperopt's "
                                                                       "fmin function.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")

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
