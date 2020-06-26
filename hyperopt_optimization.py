import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from distutils.util import strtobool
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.applications import VGG19

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
                                                      count_samples_mode=True, rgb_preprocessor=None))

    from itertools import chain, combinations

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    space = {'alpha': hp.choice('alpha', [0.3, 3.0, 5.0]),
             'learning_rate': hp.choice('learning_rate', [0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
             'train_vgg': hp.choice('train_vgg', [True, False])}

    def f_nn(params):

        vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(None, None, 3), pooling=None)
        encoder = Model(vgg19.input, vgg19.get_layer("block5_conv4").output, name="encoder")

        encoder.trainable = params["train_vgg"]

        d_i = Input(shape=(encoder.output.shape[1:]), name='decoder_input')
        block5_up = UpSampling2D(size=(2, 2), name="block5_up")(d_i)

        block4_1_conv0 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv0")(
            block5_up)
        block4_merge = Concatenate(axis=-1, name="block4_merge")(
            [vgg19.get_layer("block4_conv4").output, block4_1_conv0])
        block4_1_conv1 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv1")(
            block4_merge)
        block4_1_conv2 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv2")(
            block4_1_conv1)
        block4_1_conv3 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv3")(
            block4_1_conv2)
        block4_1_conv4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv4")(
            block4_1_conv3)
        block4_up = UpSampling2D(size=(2, 2), name="block4_up")(block4_1_conv4)

        block3_1_conv0 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv0")(
            block4_up)
        block3_merge = Concatenate(axis=-1, name="block3_merge")(
            [vgg19.get_layer("block3_conv4").output, block3_1_conv0])
        block3_1_conv1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv1")(
            block3_merge)
        block3_1_conv2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv2")(
            block3_1_conv1)
        block3_1_conv3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv3")(
            block3_1_conv2)
        block3_1_conv4 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv4")(
            block3_1_conv3)
        block3_up = UpSampling2D(size=(2, 2), name="block3_up")(block3_1_conv4)

        block2_1_conv0 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_1_conv0")(
            block3_up)
        block2_merge = Concatenate(axis=-1, name="block2_merge")(
            [vgg19.get_layer("block2_conv2").output, block2_1_conv0])
        block2_1_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_1_conv1")(
            block2_merge)
        block2_1_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_1_conv2")(
            block2_1_conv1)
        block2_up = UpSampling2D(size=(2, 2), name="block2_up")(block2_1_conv2)

        block1_1_conv0 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_1_conv0")(
            block2_up)
        block1_merge = Concatenate(axis=-1, name="block1_merge")(
            [vgg19.get_layer("block1_conv2").output, block1_1_conv0])
        block1_1_conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_1_conv1")(
            block1_merge)
        block1_1_conv2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_1_conv2")(
            block1_1_conv1)
        block1_1_conv3 = Conv2D(filters=2, kernel_size=3, padding='same', activation='relu', name="block1_1_conv3")(
            block1_1_conv2)
        output = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name="output")(
            block1_1_conv3)

        decoder = Model([vgg19.input, d_i], output, name="decoder")

        decoder_output = decoder([vgg19.input, encoder(encoder.input)])
        model = Model(encoder.input, decoder_output, name=vgg19.name)

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

        my_loss = custom_losses.bce_dsc_loss(params["alpha"])

        metrics_list = [custom_losses.dice_coef, keras_metrics.Precision(), keras_metrics.Recall()]
        model.compile(optimizer=Adam(lr=params["learning_rate"]), loss=my_loss, metrics=metrics_list
                      , experimental_run_tf_function=False
                      )

        es = EarlyStoppingAtMinValLoss(test_paths, file_path=None, patience=20, rgb_preprocessor=rgb_preprocessor)
        history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size, resize=False,
                                                         rgb_preprocessor=rgb_preprocessor),
                            epochs=args.epochs,
                            verbose=0,
                            callbacks=[es],
                            steps_per_epoch=n_train_samples // 4)

        metrics = model.evaluate(
            x=data.test_image_generator(test_paths, input_size=None, batch_size=1, rgb_preprocessor=rgb_preprocessor),
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
            data.save_results_on_paths(model, training_paths, "results_training", rgb_preprocessor=rgb_preprocessor)
            data.save_results_on_paths(model, test_paths, "results_test", rgb_preprocessor=rgb_preprocessor)

        plt.clf()
        for key in history.history.keys():
            if key in ["val_dice_coef", "dice_coef"]:
                plt.plot(history.history[key])
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

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("my_model.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), len(trials.trials),
                                                                   args.fmin_max_evals))
        global iteration
        iteration += len(trials.trials)
        args.fmin_max_evals = len(trials.trials) + args.fmin_max_evals
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=args.fmin_max_evals, trials=trials)
    print('best: ')
    print(space_eval(space, best))

    # save the trials object
    with open("my_model.hyperopt", "wb") as f:
        pickle.dump(trials, f)

    result = "Best DSC: {:.4f}\nParameters: {}\n\nDSC, Parameters\n".format(-trials.best_trial["result"]["loss"],
                                                                            space_eval(space, best))
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
