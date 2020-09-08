# deep_crack_detection
Code for testing Deep Learning models for crack detection with CrackForest [https://github.com/cuilimeng/CrackForest-dataset] dataset; Aigle-RN and ESAR (2 out of the 3 parts of the "CrackDataset")[https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html] datasets; and the cropped CRACK500, GAPs384 and cracktree200 [https://github.com/fyangneil/pavement-crack-detection] datasets.

_You should download the datasets from the corresponding links and cite the sources._

## Pre-requisites
This repository was tested on Ubuntu 18.04 with a Nvidia GeForce GTX 1050 using Driver Version 440.82 and CUDA Version 10.2. The network was build using Tensorflow 2.1.0. An environment.yml is provided in this repository to clone the environment used (recommended).

## How to run
To train and validate on CrackForest and Aigle-RN combined, for example, run:
```
python train_and_validate.py --dataset_names "cfd" "aigle-rn" --dataset_paths "path/to/cfd" "path/to/crackdataset"
```

The program will then train the default model using the listed datasets. Training images are split into training and test images using a 80/20 proportion; the model will be trained until the desired number of epochs (default: 150) or if the loss in the test images doesn't improve for 20 epochs.

After the training is done, a results_date_time folder will be created. It contains a csv file with the training history and a plot of such training history. Additionally, there are 4 folders:
* "results_training": it contains a visual comparison of ground truth and predicted cracks in the images used for training. It contains a hdf5 with the weights from the last training epoch too.
* "results_training_min_val_loss": the same as before, but using the weights from the epoch with the minimum loss in the test images during training.
* "results_test": it contains a visual comparison of ground truth and predicted cracks in the images used for testing. It contains a text file with numeric metrics of the performance in the test set.
* "results_test_min_val_loss": the same as before, but using the weights from the epoch with the minimum loss in the test images during training.

### Input arguments

Look at 'models_dict' in _train_and_validate.py_ for a full list of available models. Additional models can be added in the models folder using a Python file per model and adding it to the 'models_dict.'

Below the whole list of available input arguments:

* ("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'crack500', 'gaps384', 'cracktree200'")
* ("--dataset_paths", type=str, nargs="+",
                    help="Path to the folders containing the respective datasets as downloaded from the original source.")
* ("--model", type=str, default="uvgg19",
                    help="Network to use. It can be either a name from 'models.available_models.py' or a path to a json file.")
* ("--training_crop_size", type=int, nargs=2, default=[256, 256],
                    help="For memory efficiency and being able to admit multiple size images, subimages are created by cropping original images to this size windows")
* ("--alpha", type=float, default=0.5,
                    help="Alpha for objective function: BCE_loss + alpha*DSC_loss")
* ("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
* ("--epochs", type=int, default=150, help="Number of epochs to train.")
* ("--batch_size", type=int, default=4, help="Batch size for training.")
* ("--pretrained_weights", type=str, default=None,
                    help="Load previous weights from this location.")
* ("--use_da", type=str, default="True", help="If 'True', training will be done using data "
                                                               "augmentation. If 'False', just raw images will be "
                                                               "used.")