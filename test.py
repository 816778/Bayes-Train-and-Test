#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing module of the bnn4hi package

This module contains the main function to test the calibration of the
trained models generating the `reliability diagram`, test the accuracy
of the models with respect to the uncertainty of the predictions
generating the `uncertainty vs accuracy plot` and test the uncertainty
of each model, class by class, generating the `class uncertainty plot`
of each dataset.

This module can be imported as a part of the bnn4hi package, but it can
also be launched from command line, as a script. For that, use the `-h`
option to see the required arguments.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tensorflow.keras.utils import to_categorical
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import yaml
import torch
import sys


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Local imports
if '.' in __name__:
    # To run as a module
    from .lib import my_config
    from lib.data import load_dataset_images
    from .lib.analysis import *
    from kaggle.src.get_dataset import get_new_model
    from .lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                           plot_accuracy_vs_uncertainty, plot_model_accuracy, plot_confusion_matrix)
else:

    # To run as a script
    from lib import my_config
    from lib.data import load_dataset_images
    from lib.analysis import *
    from kaggle.src.get_dataset import get_new_model
    from lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                          plot_accuracy_vs_uncertainty, plot_model_accuracy, plot_confusion_matrix)


# PARAMETERS
# =============================================================================

def _parse_args(dataset_list):
    """Analyses the received parameters and returns them organised.

    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.

    Parameters
    ----------
    dataset_list : list of str
        List with the abbreviated names of the datasets to test. If
        `test.py` is launched as a script, the received parameters must
        correspond to the order of this list.

    Returns
    -------
    out : namespace
        The namespace with the values of the received parameters
        assigned to objects.
    """

    # Generate the parameter analyser
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    # Add arguments
    parser.add_argument("epochs",
                        type=int,
                        nargs=len(dataset_list),
                        help=("List of the epoch of the selected checkpoint "
                              "for testing each model. The order must "
                              f"correspond to: {dataset_list}."))

    # Return the analysed parameters
    return parser.parse_args()


# PREDICT FUNCTIONS
# =============================================================================

def configure_model(model_path):
    model = EfficientNet.from_pretrained('efficientnet-b1')

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model


def predict(model, X_test, y_test, verbose=True):
    # Convertir X_test de numpy.ndarray a torch.Tensor
    X_test = torch.from_numpy(X_test).float()
    X_test = X_test[:500]  # Submuestrear X_test si es necesario
    y_test = y_test[:500]  # Asegurarse de que y_test también esté submuestreado

    if verbose:
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of y_test: {y_test.shape}")
        print(f"Unique classes in y_test: {np.unique(y_test)}")

    with torch.no_grad():  # Desactivar el cálculo de gradientes para inferencia
        predictions = model(X_test)
        print("Output shape:", predictions.shape)

    y_test = torch.tensor(y_test)

    probabilities = torch.sigmoid(predictions)
    binary_predictions = (probabilities > 0.51).int()
    isup_predictions = binary_predictions[:, :5].sum(dim=1)
    final_isup_predictions = torch.round(isup_predictions).int()

    if verbose:
        print(f"Shape of final_isup_predictions: {final_isup_predictions.shape}")
        print("y_test samples:", y_test[:10])
        print("y_pred samples:", final_isup_predictions[:10])
        # print(predictions)

    # Calcular la precisión (accuracy)
    accuracy = torch.mean((final_isup_predictions == y_test).float())

    if verbose:
        print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


def custom_loss_function(y, rv_y):
    return -rv_y.log_prob(y)


# MAIN FUNCTION
# =============================================================================

def test(verbose=False):
    """Tests the trained bayesian models

    The plots are saved in the `TEST_DIR` defined in `config.py`.

    Parameters
    ----------
    epochs : dict
        Dict structure with the epochs of the selected checkpoint for
        testing each model. The keys must correspond to the abbreviated
        name of the dataset of each trained model.
    """

    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------

    # Input, output and dataset references
    d_path = my_config.DATA_PATH
    base_dir = my_config.MODELS_DIR
    datasets = my_config.DATASETS
    output_dir = my_config.TEST_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Model parameters
    l1_n = my_config.LAYER1_NEURONS
    l2_n = my_config.LAYER2_NEURONS

    # Training parameters
    p_train = my_config.P_TRAIN
    learning_rate = my_config.LEARNING_RATE

    # Bayesian passes
    passes = my_config.BAYESIAN_PASSES

    # Plot parameters
    colours = my_config.COLOURS
    w = my_config.PLOT_W
    h = my_config.PLOT_H

    # Plotting variables
    reliability_data = {}
    acc_data = {}
    px_data = {}

    # FOR EVERY DATASET
    # -------------------------------------------------------------------------

    for name, dataset in datasets.items():

        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']

        MODEL_DIR = "../kaggle/final_models"

        print("MODEL DIR: ", MODEL_DIR)
        if not os.path.isdir(MODEL_DIR):
            reliability_data[name] = []
            acc_data[name] = []
            px_data[name] = []
            print("MODELO NO ENCONTRADO")
            continue

        # GET DATA
        # ---------------------------------------------------------------------
        # Get dataset
        dir_data_path = os.path.join(my_config.DATA_PATH, 'img_data')
        _, _, _, _, X_test, y_test = load_dataset_images(dir_data_path)
        # X_test = np.squeeze(X_test)
        # LOAD MODEL
        # ---------------------------------------------------------------------
        # Load trained model
        model_path = os.path.join(MODEL_DIR, 'final_2_efficientnet-b1_kfold_5_latest.pt')
        yaml_path = os.path.join('..', 'kaggle', 'src', 'configs', 'final_2.yaml')
        model = get_new_model(model_path, yaml_path)
        # model = configure_model(model_path)

        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------

        # Tests message
        print(f"\n### Starting {name} tests")
        print('#' * 80)
        print(f"\nMODEL DIR: {model_path}")

        # Launch predictions
        accuracy = predict(model, X_test, y_test)

        # Liberate model
        del model


if __name__ == "__main__":
    test()
