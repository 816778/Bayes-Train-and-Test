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
from tensorflow import keras
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tensorflow.keras.utils import to_categorical

# Local imports
if '.' in __name__:

    # To run as a module
    from .lib import my_config
    from lib.data import get_dataset
    from .lib.analysis import *
    from .lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                           plot_accuracy_vs_uncertainty, plot_model_accuracy, plot_confusion_matrix)
else:

    # To run as a script
    from lib import my_config
    from lib.data import get_dataset
    from lib.analysis import *
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

def predict(model, X_test, y_test, samples=100, verbose=True):
    if verbose:
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of y_test: {y_test.shape}")
        print(f"Unique classes in y_test: {np.unique(y_test)}")

    print(f"\nLanzando {samples} predicciones bayesianas")
    predictions = bayesian_predictions(model, X_test, samples=samples)
    y_pred_mean = np.mean(predictions, axis=0).argmax(axis=1)
    if verbose:
        print(f"Shape of predictions: {predictions.shape}")
        print("y_pred_mean shape:", y_pred_mean.shape)

    rd_data = (reliability_diagram(predictions, y_test))

    # Cross entropy and accuracy
    print("\nGenerating data for the `accuracy vs uncertainty` plot",
          flush=True)
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)

    print("\nGenerating data for the `class uncertainty` plot", flush=True)
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)

    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep, y_pred_mean


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def custom_accuracy(y_true, y_pred):
    y_pred_rounded = tf.math.round(y_pred)
    correct_predictions = tf.reduce_all(tf.equal(y_true, y_pred_rounded), axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


# MAIN FUNCTION
# =============================================================================

def test(epochs, verbose=True):
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

        # Get model dir
        model_dir = (f"{name}_{l1_n}-{l2_n}model_{p_train}train"
                     f"_{learning_rate}lr")
        model_dir = os.path.join(model_dir, f"epoch_{epochs[name]}")
        model_dir = os.path.join(base_dir, model_dir)

        print("MODEL DIR: ", model_dir)
        if not os.path.isdir(model_dir):
            reliability_data[name] = []
            acc_data[name] = []
            px_data[name] = []
            print("MODELO NO ENCONTRADO")
            exit()

        # GET DATA
        # ---------------------------------------------------------------------
        # Get dataset
        _, _, _, _, X_test, y_test = get_dataset('args.data_path', 'args.csv_path', 6)
        X_test = np.squeeze(X_test)
        print(f'X_test.shape: {X_test.shape}\n')
        # LOAD MODEL
        # ---------------------------------------------------------------------
        # Load trained model
        model = tf.keras.models.load_model(model_dir)

        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------

        # Tests message
        print(f"\n### Starting {name} tests")
        print('#' * 80)
        print(f"\nMODEL DIR: {model_dir}")

        # Launch predictions
        (reliability_data[name],
         acc_data[name],
         px_data[name],
         avg_Ep, avg_H_Ep, y_pred_mean) = predict(model, X_test, y_test, samples=passes)

        # Obtener las predicciones del modelo
        if verbose:
            print("y_test samples:", y_test[:10])
            print("y_pred samples:", y_pred_mean[:10])

        # Generar la matriz de confusión
        classes = [f'Class {i}' for i in range(num_classes)]
        plot_confusion_matrix(y_test, y_pred_mean, classes, output_dir, name, normalize=True)
        # Liberate model
        del model

        # IMAGE-RELATED PLOTS
        # ---------------------------------------------------------------------
        # Plot class uncertainty
        plot_class_uncertainty(output_dir, name, epochs[name], avg_Ep, avg_H_Ep, w, h)

    # End of tests message
    print("\n### Tests finished")
    print('#' * 80, flush=True)

    # Generate accuracy plot
    plot_model_accuracy(acc_data, output_dir)

    # GROUPED PLOTS
    # -------------------------------------------------------------------------
    plot_reliability_diagram(output_dir, reliability_data, w, h, colours)

    # Plot accuracy vs uncertainty
    plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours)

if __name__ == "__main__":

    # Parse args
    dataset_list = my_config.DATASETS_LIST
    args = _parse_args(dataset_list)

    # Generate parameter structures for main function
    epochs = {}
    for i, name in enumerate(dataset_list):
        epochs[name] = args.epochs[i]

    # Launch main function
    test(epochs)
