#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data module of the bnn4hi package

The functions of this module are used to load, preprocess and organise
data from hyperspectral datasets.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
import csv
from scipy import io
from numpy.random import randint as rand
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import pickle
import sys
import pandas as pd
import ast
import torch
import pandas as pd


# DATA FUNCTIONS
# =============================================================================
def _load_image(file_path='/data/hook/feature_activations.npy', csv_path='/data/train.csv', is_torch=True):
    """Loads the image data and the ground truth from CSV files.

    Parameters
    ----------
    data_path : str
        Path of the CSV file containing the image data.
    label_file_path : str
        Path of the CSV file containing the ground truth labels.

    Returns
    -------
    X : ndarray
        Image data.
    y : ndarray
        Ground truth labels.
    """
    csv.field_size_limit(sys.maxsize)
    df = pd.read_csv(csv_path)  # train  train_prueba

    image_names = df['image_id']
    y = df['isup_grade']
    activations = np.load(file_path)

    if is_torch:
        activations = torch.tensor(activations, dtype=torch.float32)

    print(f"Imágenes cargadas: {len(image_names)}\n")
    return np.array(image_names), np.array(activations), np.array(y)


def preprocess_data(X):
    # Aplanar los datos de entrada
    X = X.reshape(X.shape[0], -1)
    # Asegurarse de que los datos sean del tipo float32
    X = X.astype(np.float32)
    return X



def _standardise(X):
    """Standardises a set of hyperspectral pixels
    
    Parameters
    ----------
    X : ndarray
        Set of hyperspectral pixels.
    
    Returns
    -------
    X_standardised : ndarray
        The received set of pixels standardised.
    """

    return (X - X.mean(axis=0)) / X.std(axis=0)


def _normalise(X):
    """Normalises a set of hyperspectral pixels
    
    Parameters
    ----------
    X : ndarray
        Set of hyperspectral pixels.
    
    Returns
    -------
    X_normalised : ndarray
        The received set of pixels normalised.
    """

    X -= X.min()
    return X / X.max()


def _preprocess(X, y, standardisation=False, only_labelled=True):
    """Preprocesses the input data by optionally standardising it.
    Parameters
    ----------
    X : ndarray
        Input data, expected in the shape (n_samples, n_features).
    y : ndarray, optional
        Labels corresponding to `X`. Can be used for additional processing.
    standardisation : bool, optional (default: False)
        Flag to activate standardisation.
    standardistion : bool, optional (default: False)
        Flag to activate standardisation.
    only_labelled : bool, optional (default: True)
        Flag to remove unlabelled pixels.
    
    Returns
    -------
    X : ndarray
        Preprocessed data.
    y : ndarray, optional
        Labels if provided.
    """
    # Apply standardisation if requested
    if standardisation:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X


# GET DATASET FUNCTION
# =============================================================================

def get_dataset(file_path, csv_path, seed=35, return_names = False):
    """Returns the preprocessed training and testing data and labels
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    X_val : ndarray
        Validation data set.
    y_val : ndarray
        Validation data set labels.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    """

    # Load image
    image_names, X, y = _load_image(file_path, csv_path)

    # Separate into train and test data sets
    X_train, X_temp, y_train, y_temp, names_train, names_temp = train_test_split(
        X, y, image_names, test_size=0.3, random_state=seed)

    X_val, X_test, y_val, y_test, names_val, names_test = train_test_split(
        X_temp, y_temp, names_temp, test_size=0.5, random_state=seed)
    

    if return_names:
        return X_train, y_train, X_val, y_val, X_test, y_test, names_train, names_val, names_test
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test


def get_filtered_dataset(file_path, csv_path, csv_file_uncertanty, threshold, seed=35, return_names=False):
    """
    Filtra las imágenes con incertidumbre menor o igual al umbral indicado
    DESPUÉS de dividir los datos en entrenamiento, validación y prueba.

    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    csv_file : str
        Ruta al archivo CSV que contiene las incertidumbres y nombres de imágenes.
    threshold : float
        Umbral de incertidumbre. Solo se incluirán imágenes con incertidumbre <= threshold.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data.
    return_names : bool, optional (default: False)
        Si es True, también devuelve los nombres de las imágenes.

    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    X_val : ndarray
        Validation data set.
    y_val : ndarray
        Validation data set labels.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    (Opcional) names_train, names_val, names_test : list
        Listas con los nombres de las imágenes en cada conjunto.
    """

    # Load image data
    image_names, X, y = _load_image(file_path, csv_path)
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    print(f"Unique classes in y_test: {np.unique(y)}")

    # Load uncertainty data from CSV
    uncertainty_df = pd.read_csv(csv_file_uncertanty, delimiter=";")
    print(f"Loaded uncertainty data from {csv_file_uncertanty}")

    # Map uncertainties to image names for filtering later
    uncertainty_map = dict(zip(uncertainty_df["Image Name"], uncertainty_df["Uncertainty"]))

    # Separate into train and test data sets
    X_train, X_temp, y_train, y_temp, names_train, names_temp = train_test_split(
        X, y, image_names, test_size=0.3, random_state=seed)

    # Split the remaining 30% equally into validation and testing (15% each)
    X_val, X_test, y_val, y_test, names_val, names_test = train_test_split(
        X_temp, y_temp, names_temp, test_size=0.5, random_state=seed)

    # Filter each set based on the uncertainty threshold
    def filter_by_uncertainty(X, y, names):
        filtered_indices = [
            i for i, name in enumerate(names) if uncertainty_map.get(name, float("inf")) <= threshold
        ]
        X_filtered = X[filtered_indices]
        y_filtered = y[filtered_indices]
        names_filtered = [names[i] for i in filtered_indices]
        return X_filtered, y_filtered, names_filtered

    X_train, y_train, names_train = filter_by_uncertainty(X_train, y_train, names_train)
    X_val, y_val, names_val = filter_by_uncertainty(X_val, y_val, names_val)
    X_test, y_test, names_test = filter_by_uncertainty(X_test, y_test, names_test)

    # Imprimir tamaño de sets
    print(f"X_train Shape: {X_train.shape}")
    print(f"y_train Shape: {y_train.shape}")
    print(f"X_val Shape: {X_val.shape}")
    print(f"y_val Shape: {y_val.shape}")
    print(f"X_test Shape: {X_test.shape}")
    print(f"y_test Shape: {y_test.shape}")

    # Return datasets
    if return_names:
        return X_train, y_train, X_val, y_val, X_test, y_test, names_train, names_val, names_test
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test



def get_filtered_dataset_after(dataset, data_path, p_train, csv_file, threshold, seed=35, return_names=False):
    """
    Filtra las imágenes con incertidumbre menor o igual al umbral indicado y devuelve los conjuntos de datos preprocesados.

    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    csv_file : str
        Ruta al archivo CSV que contiene las incertidumbres y nombres de imágenes.
    threshold : float
        Umbral de incertidumbre. Solo se incluirán imágenes con incertidumbre <= threshold.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data.
    return_names : bool, optional (default: False)
        Si es True, también devuelve los nombres de las imágenes.

    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    X_val : ndarray
        Validation data set.
    y_val : ndarray
        Validation data set labels.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    (Opcional) names_train, names_val, names_test : list
        Listas con los nombres de las imágenes en cada conjunto.
    """

    # Load image data
    image_names, X, y = _load_image()
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    print(f"Unique classes in y_test: {np.unique(y)}")

    # Load uncertainty data from CSV
    uncertainty_df = pd.read_csv(csv_file, delimiter=";")
    print(f"Loaded uncertainty data from {csv_file}")

    # Filter images based on the uncertainty threshold
    filtered_df = uncertainty_df[uncertainty_df["Uncertainty"] <= threshold]
    filtered_image_names = filtered_df["Image Name"].tolist()
    print(f"Number of images after filtering by threshold {threshold}: {len(filtered_image_names)}")

    # Filter the image data based on the filtered names
    mask = [name in filtered_image_names for name in image_names]
    X_filtered = X[mask]
    y_filtered = y[mask]
    image_names_filtered = [name for name, keep in zip(image_names, mask) if keep]

    # Separate into train and test data sets
    X_train, X_temp, y_train, y_temp, names_train, names_temp = train_test_split(
        X_filtered, y_filtered, image_names_filtered, test_size=0.3, random_state=seed)

    # Split the remaining 30% equally into validation and testing (15% each)
    X_val, X_test, y_val, y_test, names_val, names_test = train_test_split(
        X_temp, y_temp, names_temp, test_size=0.5, random_state=seed)

    # Return datasets
    if return_names:
        return X_train, y_train, X_val, y_val, X_test, y_test, names_train, names_val, names_test
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test

# NOISY DATASET FUNCTIONS
# =============================================================================

def _generic_noise(shape, p_noise):
    """Generates an array of `int16` random value variations
    
    Parameters
    ----------
    shape : tuple of ints
        The shape of the expected array.
    p_noise : float
        Represents, from 0.0 to 1.0, the noise factor. The closest to
        1.0 will produce the higher noise values.
    
    Returns
    -------
    variations : ndarray
        Array of random variations within the values range of `int16`
        and proportional to `p_noise`.
    """

    return rand(-2 ** 15, 2 ** 15 - 1, size=shape, dtype='int16') * p_noise


def get_noisy_dataset(dataset, data_path, p_train, noises, seed=35):
    """Returns train set and several test sets with increasing noise
    
    Parameters
    ----------
    dataset: dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path: str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train: float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    noises : array_like of floats
        Each value represents, from 0.0 to 1.0, one noise factor. The
        closest to 1.0 will produce the higher noise values. A testing
        set will be generated for each received noise.
    seed: int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    noisy_X_tests : ndarray
        One testing data set per received noise.
    y_test : ndarray
        Testing data set labels. It is the same for every testing set.
    """

    # Load image
    image_names, X, y = _load_image(dataset, data_path)

    # Noise preprocessing settings (to standardise after generating the noise)
    X_mean = X.reshape(-1, X.shape[2]).mean(axis=0)
    X_std = X.reshape(-1, X.shape[2]).std(axis=0)

    # Preprocess
    X, y = _preprocess(X, y, standardisation=False)

    # Separate into train, val and test data sets
    p_test = 1 - p_train
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=p_test,
                                         random_state=seed, stratify=y)

    noisy_X_tests = []
    for noise in noises:
        # Add noise to `X_test` and append it to `noisy_X_tests`
        noisy_X_tests.append(X_test + _generic_noise(X_test.shape, noise))

    # Standardise `X_train` and each `noisy_X_tests`
    X_train = (X_train - X_mean) / X_std
    for n, X_test in enumerate(noisy_X_tests):
        noisy_X_tests[n] = (X_test - X_mean) / X_std

    # Return train and test sets
    return X_train, y_train, noisy_X_tests, y_test


# MIXED DATASET FUNCTIONS
# =============================================================================

def _mix_classes(y_train, class_a, class_b):
    """Mixes the labels between two classes
    
    It does not return, the `y_train` array is modified in-place.
    
    Parameters
    ----------
    y_train : ndarray
        Training data set labels to be modified.
    class_a : int
        Number of the first class to be mixed.
    class_b : int
        Number of the second class to be mixed.
    """

    # Get the indices of the pixels from both classes
    index = (y_train == class_a) | (y_train == class_b)

    # Estract and shuffle their values
    values = y_train[index]
    np.random.shuffle(values)

    # Modify the original values with the new ones
    y_train[index] = values


def get_mixed_dataset(dataset, data_path, p_train, class_a, class_b, seed=35):
    """Returns the datasets with mixed classes on the training set
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    class_a : int
        Number of the first class to be mixed.
    class_b : int
        Number of the second class to be mixed.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels with mixed labels for `class_a` and
        `class_b`.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    """

    # Get dataset
    X_train, y_train, X_test, y_test = get_dataset(dataset, data_path, p_train,
                                                   seed=seed)

    # Mix the labels between two classes
    _mix_classes(y_train, class_a, class_b)

    return X_train, y_train, X_test, y_test


# MAP FUNCTIONS
# =============================================================================

def get_map(dataset, data_path):
    """Returns all the pixels and labels of the image preprocessed
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    
    Returns
    -------
    X : ndarray
        Hyperspectral image pixels standardised.
    y : ndarray
        Ground truth.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    """

    # Load image
    X, y = _load_image(dataset, data_path)
    shape = y.shape

    # Preprocess
    X, y = _preprocess(X, y, standardisation=True, only_labelled=False)

    return X, y, shape


# IMAGE FUNCTIONS (FOR RGB REPRESENTATION)
# =============================================================================

def get_image(dataset, data_path):
    """Returns the image prepared for the RGB representation algorithm
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    
    Returns
    -------
    X : ndarray
        Hyperspectral image pixels normalised.
    shape : tuple of ints
        Original shape to reconstruct the image.
    """

    # Load image
    X, y = _load_image(dataset, data_path)
    shape = X.shape

    # Preprocess
    X, _ = _preprocess(X, y, standardisation=False, only_labelled=False)

    # Normalise
    X = _normalise(X)

    return X, shape
