import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import custom_losses
import data

from unet import unet
from multiscale_unet import multiscale_unet

models_dict = {
    "unet": unet,
    "multiscale_unet": multiscale_unet
}


def main(args):
    input_size = (256, 256)
    model = models_dict[args.model]((input_size[0], input_size[1], 1))
    if args.pretrained_weights:
        model.load_weights(args.pretrained_weights)
    model.compile(optimizer=Adam(lr=1e-4), loss=custom_losses.loss(args.alpha), metrics=[custom_losses.dice_coef,
                                                                                         'binary_crossentropy'])

    model_checkpoint = ModelCheckpoint('%s_best.hdf5' % args.model, monitor='loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)

    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    n_training_images = int(0.8 * paths.shape[1])
    np.random.seed(0)
    np.random.shuffle(paths.transpose())
    training_paths = paths[:, :n_training_images]
    test_paths = paths[:, n_training_images:]
    history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size), epochs=args.epochs,
                        verbose=2, callbacks=[model_checkpoint, es],
                        steps_per_epoch=training_paths.shape[1] // args.batch_size)
    model.save_weights("%s.hdf5" % args.model)

    data.save_results_on_paths(model, training_paths, "results_training")
    data.save_results_on_paths(model, test_paths, "results_test")
    metrics = model.evaluate(x=data.train_image_generator(test_paths, input_size, 1), steps=test_paths.shape[1])
    result_string = ""
    for idx, metric in enumerate(model.metrics_names):
        result_string += "{}: {:.4f}\n".format(metric, metrics[idx])
    with open(os.path.join("results_test", "results.txt"), "w") as f:
        f.write(result_string.strip())

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['binary_crossentropy'])
    plt.ylim((0.0, 1.0 + args.alpha))
    plt.title('model losses')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['loss', 'dice_coef', 'binary_crossentropy'], loc='upper left')
    plt.savefig("training_losses.png")
    # plt.show()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folders containing the datasets as downloaded from the original source.")
    parser.add_argument("--model", type=str, default="unet", help="Network to use.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for loss BCE + alpha*DSCloss")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Load previous weights from this location.")

    args_dict = parser.parse_args(args)
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)

