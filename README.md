# deep_crack_detection
Code for testing Deep Learning models for crack detection with CrackForest Dataset, and the Aigle-RN and ESAR datasets.

## Pre-requisites
This repository was tested on Ubuntu 18.04 with a Nvidia GeForce GTX 1050 using Driver Version 440.82 and CUDA Version 10.2. The network was build using Tensorflow 2.1.0. An environment.yml is provided in this repository to clone the environment used. The packages installed in this environment were:
* python 3.7
* opencv 3.4.2
* scipy 1.4.1
* matplotlib 3.1.3
* tensorflow (tensorflow-gpu) 2.1.0

## How to run
To train and validate on CrackForest and Aigle-RN combined, for example, run:
```
python train_and_validate.py --dataset_names "cfd" "aigle-rn" --dataset_paths "path/to/cfd" "path/to/aigle"
```

Below the whole list of available input arguments:

* ("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
* ("--dataset_paths", type=str, nargs="+",
                    help="Path to the folders containing the datasets as downloaded from the original source.")
* ("--model", type=str, default="unet", help="Network to use.")
* ("--alpha", type=float, default=0.5, help="Alpha for loss BCE + alpha*DSCloss")
* ("--epochs", type=int, default=150, help="Number of epochs to train.")
* ("--batch_size", type=int, default=4, help="Batch size for training.")
* ("--pretrained_weights", type=str, default=None,
                    help="Load previous weights from this location.")
