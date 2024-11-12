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

# Local imports
if '.' in __name__:

    # To run as a module
    from .lib import my_config
    from lib.data import read_test_dataset
    from .lib.analysis import *
    from .lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                           plot_accuracy_vs_uncertainty, plot_model_accuracy, plot_confusion_matrix, plot_combined_noise)
else:

    # To run as a script
    from lib import my_config
    from lib.data import read_test_dataset
    from lib.analysis import *
    from lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                          plot_accuracy_vs_uncertainty, plot_model_accuracy, plot_confusion_matrix, plot_combined_noise)


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
    parser.add_argument("numpy_file_path", help="Ruta al fichero de datos de test")

    # Return the analysed parameters
    return parser.parse_args()


def load_all_test_datasets(main_directory):
    """
    Recorre un directorio general, busca subdirectorios y carga todos los ficheros .npz encontrados.

    Parameters
    ----------
    main_directory : str
        Ruta del directorio principal que contiene los subdirectorios con los archivos .npz.

    Returns
    -------
    all_X_tests : list of ndarray
        Lista que contiene todos los arrays X_test encontrados en los archivos .npz.
    all_y_tests : list of ndarray
        Lista que contiene todos los arrays y_test encontrados en los archivos .npz.
    """
    all_X_tests = []
    all_y_tests = []

    # Recorre el directorio principal
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                X_test, y_test = read_test_dataset(file_path)
                all_X_tests.append(X_test)
                all_y_tests.append(y_test)

    return all_X_tests, all_y_tests


# PREDICT FUNCTIONS
# =============================================================================

def predict(model, X_test, y_test, samples=100, verbose=True):
    if verbose:
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of y_test: {y_test.shape}")
        print(f"Unique classes in y_test: {np.unique(y_test)}")

    print(f"\nLanzando {samples} predicciones bayesianas")
    predictions = bayesian_predictions(model, X_test, samples=samples)
    if verbose:
        print(f"Shape of predictions: {predictions.shape}")

    rd_data = (reliability_diagram(predictions, y_test))

    # Cross entropy and accuracy
    print("\nGenerating data for the `accuracy vs uncertainty` plot",
          flush=True)
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)

    print("\nGenerating data for the `class uncertainty` plot", flush=True)
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)

    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep


def predict_2(model, X_test, y_test, samples=100):
    """Launches the bayesian predictions

    Launches the necessary predictions over `model` to collect the data
    to generate the `reliability diagram`, the `uncertainty vs accuracy
    plot` and the `class uncertainty` plot of the model.

    To generate the `reliability diagram` the predictions are divided
    into groups according to their predicted probability. To generate
    the `uncertainty vs accuracy` plot the predictions are divided into
    groups according to their uncertainty value. For that, it uses the
    default number of groups defined in the `reliability_diagram` and
    the `accuracy_vs_uncertainty` functions of `analysis.py`.

    Parameters
    ----------
    model : TensorFlow Keras Sequential
        The trained model.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.

    Returns
    -------
    rd_data : list of float
        List of the observed probabilities of each one of the predicted
        probability groups.
    acc_data : list of float
        List of the accuracies of each one of the uncertainty groups.
    px_data : list of float
        List of the percentage of pixels belonging to each one of the
        uncertainty groups.
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    avg_H_Ep : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    """

    # Bayesian stochastic passes
    print(f"\nLaunching {samples} bayesian predictions")
    predictions = bayesian_predictions(model, X_test, samples=samples)

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    print(f"Unique classes in y_test: {np.unique(y_test)}")
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Predictions: {predictions[0]}")

    # Reliability Diagram
    print("\nGenerating data for the `reliability diagram`", flush=True)
    rd_data = reliability_diagram(predictions, y_test)
    # Cross entropy and accuracy
    print("\nGenerating data for the `accuracy vs uncertainty` plot",
          flush=True)
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)

    # Analyse entropy
    print("\nGenerating data for the `class uncertainty` plot", flush=True)
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)

    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep


def noise_predict(model, X_test, y_test, samples=100):
    """Launches the bayesian noise predictions

    Launches the necessary predictions over `model` to collect the data
    to generate the `combined noise` plot.

    Parameters
    ----------
    model : TensorFlow Keras Sequential
        The trained model.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.

    Returns
    -------
    avg_H : ndarray
        List of the averages of the global uncertainty (H) of each
        class. The last position also contains the average of the
        entire image.
    """

    # Bayesian stochastic passes
    print(f"\nLaunching {samples} bayesian predictions")
    predictions = bayesian_predictions(model, X_test, samples=samples)

    # Analyse entropy
    avg_H, _, _ = analyse_entropy(predictions, y_test)

    return avg_H


def custom_loss_function(y, rv_y):
    return -rv_y.log_prob(y)

# MAIN FUNCTION
# =============================================================================

def test(epochs, numpy_dir_path, verbose=False):
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

    # Noise testing parameters
    noises = my_config.NOISES

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

    data = {}

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
            data[name] = []
            print("MODELO NO ENCONTRADO")
            continue

        # GET DATA
        # ---------------------------------------------------------------------
        # Get dataset
        all_X_tests, all_y_tests = load_all_test_datasets(numpy_dir_path)
        # LOAD MODEL
        # ---------------------------------------------------------------------
        # Load trained model
        # model = tf.keras.models.load_model(model_dir, custom_objects={'<lambda>': custom_loss_function})
        model = tf.keras.models.load_model(model_dir)

        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------

        # Tests message
        print(f"\n### Starting {name} tests")
        print('#' * 80)
        print(f"\nMODEL DIR: {model_dir}")

        # Launch predictions for every noisy dataset
        noise_data = [[] for i in range(num_classes + 1)]

        for n, n_X_test in enumerate(all_X_tests):
            # Test message
            print(f"\n# Noise test {n+1} of {len(all_X_tests)}")

            X_train = np.squeeze(all_X_tests[n])
            print(f'X_test.shape: {X_train.shape}\n')
            print(f'y_test.shape: {all_y_tests[n].shape}\n')

            # Launch prediction
            avg_H = noise_predict(model, X_train, all_y_tests[n],
                                  samples=passes)
            noise_data = np.append(noise_data, avg_H[np.newaxis].T, 1)

            np.save(os.path.join(model_dir, "test_noise"), noise_data)

        # Add normalised average to data structure
        max_H = np.log(num_classes)
        data[name] = noise_data[-1] / max_H

        if len(noises) != len(data[name]):
            print(
                f"Error: La longitud de 'labels' (noises) ({len(noises)}) y 'd' (data) ({len(data[name])}) no coinciden para '{name}'")
            print(f"noises: {noises}")
            print(f"data[{name}]: {data[name]}")

        # Plot combined noise
        plot_combined_noise(output_dir, noises, data, w, h, colours)

        exit()
        # Launch predictions
        (reliability_data[name],
         acc_data[name],
         px_data[name],
         avg_Ep, avg_H_Ep) = predict(model, X_test, y_test, samples=passes)

        # Obtener las predicciones del modelo
        y_pred = model.predict(X_test).argmax(axis=1)

        if verbose:
            print("y_test samples:", y_test[:10])
            print("y_pred samples:", y_pred[:10])

        # Generar la matriz de confusión
        classes = [f'Class {i}' for i in range(num_classes)]  # Puedes ajustar esto con los nombres de tus clases
        plot_confusion_matrix(y_test, y_pred, classes, output_dir, name, normalize=True)

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
    test(epochs, args.numpy_file_path)
