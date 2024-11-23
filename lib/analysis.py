#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analysis module of the bnn4hi package

The functions of this module are used to generate and analyse bayesian
predictions.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import csv
import os
import sys
import math
import numpy as np
import tensorflow as tf
import torch
import pandas as pd

# UNCERTAINTY FUNCTIONS
#     Global uncertainty (H) corresponds to predictive entropy
#     Aleatoric uncertainty (Ep) corresponds to expected entropy
#     Epistemic uncertainty corresponds to H - Ep subtraction
# =============================================================================

def _predictive_entropy(predictions, eps=1e-10):
    """Calculates the predictive entropy of `predictions`

    The predictive entropy corresponds to the global uncertainty (H).
    The correspondent equation can be found in the paper `Bayesian
    Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty
    Metrics`.

    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.

    Returns
    -------
    pred_h : ndarray
        Predictive entropy, i.e. global uncertainty, of `predictions`
    """

    # Get number of pixels and classes
    _, num_samples, num_classes = predictions.shape # (100, 1593, 6)

    # Application of the predictive entropy equation
    entropy = np.zeros(num_samples)
    for p in range(num_samples):
        for c in range(num_classes):
            avg = np.mean(predictions[..., p, c])

            if avg > eps:
                entropy[p] += avg * math.log(avg)
            else:
                entropy[p] += eps * math.log(eps)
    return -1 * entropy

def _expected_entropy(predictions, eps=1e-10):
    """Calculates the expected entropy of `predictions`
    
    The expected entropy corresponds to the aleatoric uncertainty (Ep).
    The correspondent equation can be found in the paper `Bayesian
    Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty
    Metrics`.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_ep : ndarray
        Expected entropy, i.e. aleatoric uncertainty, of `predictions`
    """
    
    # Get number of bayesian passes, pixels and classes
    num_tests, num_pixels, num_classes = predictions.shape
    
    # Application of the expected entropy equation
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for t in range(num_tests):
            class_sum = 0
            for c in range(num_classes):
                val = predictions[t][p][c]
                if val > eps:
                    class_sum += val * math.log(val)
                else:
                    class_sum += eps * math.log(eps)
            entropy[p] -= class_sum
    
    return entropy/num_tests


def collect_false_negative_distribution_all_classes(predictions, y_test, y_pred, output_dir="./Test", num_classes=6, num_bins=10):
    """
    Genera una tabla de distribución de clases verdaderas para cada clase predicha en diferentes rangos de incertidumbre.

    Parameters
    ----------
    predictions : ndarray
        Array con las predicciones bayesianas.
    y_test : ndarray
        Etiquetas verdaderas del conjunto de prueba.
    y_pred : ndarray
        Etiquetas predichas por el modelo.
    num_classes : int, opcional
        Número total de clases (default: 6).
    num_bins : int, opcional
        Número de intervalos para la entropía (default: 10).

    Returns
    -------
    dict
        Un diccionario con un DataFrame para cada clase predicha que muestra la distribución de clases verdaderas
        en cada rango de incertidumbre.
    """

    # Calcular la entropía predictiva para cada predicción
    model_H = _predictive_entropy(predictions)  # Usa tu función de entropía predictiva existente

    # Definir los rangos de incertidumbre (entropía)
    uncertainty_bins = np.linspace(0.0, 1.0, num_bins + 1)

    # Diccionario para almacenar distribuciones por cada clase predicha
    distributions_by_predicted_class = {}

    for pred_class in range(num_classes):
        # Crear una estructura para almacenar las frecuencias de cada clase verdadera en cada rango de incertidumbre
        distribution = {f"{uncertainty_bins[i]:.1f}-{uncertainty_bins[i + 1]:.1f}": {true_class: 0 for true_class in range(num_classes)} for i in range(num_bins)}

        # Recopilar datos de falsos negativos para la clase predicha actual
        for H, true_label, pred_label in zip(model_H, y_test, y_pred):
            true_label = int(true_label)
            pred_label = int(pred_label)

            # Solo considerar falsos negativos para la clase predicha actual
            if pred_label == pred_class and true_label != pred_class:
                # Encontrar el rango de incertidumbre, limitando el índice a num_bins - 1 para evitar el error
                bin_index = min(np.digitize(H, uncertainty_bins) - 1, num_bins - 1)
                bin_label = f"{uncertainty_bins[bin_index]:.1f}-{uncertainty_bins[bin_index + 1]:.1f}"
                distribution[bin_label][true_label] += 1

        # Convertir a DataFrame y calcular el porcentaje de ocurrencias en cada rango de incertidumbre
        df = pd.DataFrame(distribution).T
        df = df.div(df.sum(axis=1), axis=0).fillna(0) * 100  # Convertir a porcentaje y llenar NaN con 0
        distributions_by_predicted_class[pred_class] = df
    
    csv_file = os.path.join(output_dir, f"distributions_by_predicted_class.csv")

    with open(csv_file, mode='w') as file:
        writer = csv.writer(file)
        for key, value in distributions_by_predicted_class.items():
            writer.writerow([key, value])

    return distributions_by_predicted_class


def calculate_uncertainty_with_labels(predictions, image_names, y_test, y_pred, output_dir, uncertainty_type="predictive", correct_factor=0.8, incorrect_factor=1.2):
    """
    Calcula la incertidumbre para cada entrada de un conjunto de datos, considerando si las predicciones son correctas o no,
    y la guarda en un archivo CSV.

    Parameters
    ----------
    predictions : ndarray
        Predicciones bayesianas generadas por el modelo, de forma (samples, num_inputs, num_classes).
    image_names : list of str
        Lista con los nombres de las imágenes asociadas a cada entrada.
    y_test : ndarray
        Etiquetas verdaderas del conjunto de datos.
    y_pred : ndarray
        Etiquetas predichas por el modelo.
    output_dir : str
        Directorio donde se guardará el archivo CSV con los resultados.
    uncertainty_type : str, opcional
        Tipo de incertidumbre que se calculará ("predictive", "aleatoric" o "epistemic").
    correct_factor : float, opcional
        Factor para ponderar la incertidumbre cuando la predicción es correcta (default: 0.8).
    incorrect_factor : float, opcional
        Factor para ponderar la incertidumbre cuando la predicción es incorrecta (default: 1.2).

    Returns
    -------
    uncertainties : ndarray
        Vector con la incertidumbre ponderada de cada entrada (longitud igual al número de entradas).
    """
    # Calcular las métricas de incertidumbre
    model_H = _predictive_entropy(predictions)  # Entropía predictiva
    model_Ep = _expected_entropy(predictions)   # Entropía esperada (aleatoria)
    model_H_Ep = model_H - model_Ep             # Incertidumbre epistémica

    # Seleccionar el tipo de incertidumbre
    if uncertainty_type == "predictive":
        uncertainties = model_H
    elif uncertainty_type == "aleatoric":
        uncertainties = model_Ep
    elif uncertainty_type == "epistemic":
        uncertainties = model_H_Ep
    else:
        raise ValueError("El tipo de incertidumbre debe ser 'predictive', 'aleatoric' o 'epistemic'")

    # Ponderar la incertidumbre en función de si la predicción es correcta o no
    uncertainties_weighted = []
    for uncertainty, true_label, pred_label in zip(uncertainties, y_test, y_pred):
        if pred_label == true_label:  # Predicción correcta
            weighted_uncertainty = uncertainty * correct_factor
        else:  # Predicción incorrecta
            weighted_uncertainty = uncertainty * incorrect_factor
        uncertainties_weighted.append(weighted_uncertainty)

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"image_uncertainties_{uncertainty_type}_weighted.csv")

    # Guardar los resultados en un archivo CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Image Name", "Uncertainty"])  # Encabezados
        for image_name, true_label, pred_label, uncertainty in zip(image_names, y_test, y_pred, uncertainties_weighted):
            writer.writerow([image_name, uncertainty])

    print(f"Incertidumbres ponderadas guardadas en {csv_file}")
    return np.array(uncertainties_weighted)



def calculate_uncertainty_for_inputs(predictions, image_names, output_dir, uncertainty_type="predictive"):
    """
    Calcula la incertidumbre para cada entrada de un conjunto de datos y la guarda en un archivo CSV.

    Parameters
    ----------
    predictions : ndarray
        Predicciones bayesianas generadas por el modelo, de forma (samples, num_inputs, num_classes).
    image_names : list of str
        Lista con los nombres de las imágenes asociadas a cada entrada.
    output_dir : str
        Directorio donde se guardará el archivo CSV con los resultados.
    uncertainty_type : str, opcional
        Tipo de incertidumbre que se calculará ("predictive", "aleatoric" o "epistemic").

    Returns
    -------
    uncertainties : ndarray
        Vector con la incertidumbre de cada entrada (longitud igual al número de entradas).
    """
    # Calcular las métricas de incertidumbre
    model_H = _predictive_entropy(predictions)  # Entropía predictiva
    model_Ep = _expected_entropy(predictions)   # Entropía esperada (aleatoria)
    model_H_Ep = model_H - model_Ep             # Incertidumbre epistémica

    # Seleccionar el tipo de incertidumbre
    if uncertainty_type == "predictive":
        uncertainties = model_H
    elif uncertainty_type == "aleatoric":
        uncertainties = model_Ep
    elif uncertainty_type == "epistemic":
        uncertainties = model_H_Ep
    else:
        raise ValueError("El tipo de incertidumbre debe ser 'predictive', 'aleatoric' o 'epistemic'")

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"image_uncertainties_{uncertainty_type}.csv")

    # Guardar los resultados en un archivo CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Image Name", "Uncertainty"])  # Encabezados
        for image_name, uncertainty in zip(image_names, uncertainties):
            writer.writerow([image_name, uncertainty])

    print(f"Incertidumbres guardadas en {csv_file}")
    return uncertainties


def collect_distribution_all_classes(predictions, y_test, y_pred, output_dir="./Test", num_classes=6, num_bins=10):
    """
    Genera una tabla de distribución de clases verdaderas para cada clase predicha en diferentes rangos de incertidumbre,
    considerando todas las predicciones (no solo los falsos negativos).

    Parameters
    ----------
    predictions : ndarray
        Array con las predicciones bayesianas.
    y_test : ndarray
        Etiquetas verdaderas del conjunto de prueba.
    y_pred : ndarray
        Etiquetas predichas por el modelo.
    output_dir : str, opcional
        Directorio donde se guardará el archivo CSV.
    num_classes : int, opcional
        Número total de clases (default: 6).
    num_bins : int, opcional
        Número de intervalos para la entropía (default: 10).

    Returns
    -------
    dict
        Un diccionario con un DataFrame para cada clase predicha que muestra la distribución de clases verdaderas
        en cada rango de incertidumbre.
    """

    # Calcular la entropía predictiva para cada predicción
    model_H = _predictive_entropy(predictions)  # Usa tu función de entropía predictiva existente

    # Definir los rangos de incertidumbre (entropía)
    uncertainty_bins = np.linspace(0.0, 1.0, num_bins + 1)

    # Diccionario para almacenar distribuciones por cada clase predicha
    distributions_by_predicted_class = {}

    for pred_class in range(num_classes):
        # Crear una estructura para almacenar las frecuencias de cada clase verdadera en cada rango de incertidumbre
        distribution = {f"{uncertainty_bins[i]:.1f}-{uncertainty_bins[i + 1]:.1f}": {true_class: 0 for true_class in range(num_classes)} for i in range(num_bins)}
        counts_per_bin = {f"{uncertainty_bins[i]:.1f}-{uncertainty_bins[i + 1]:.1f}": 0 for i in range(num_bins)}

        # Recopilar datos para la clase predicha actual
        for H, true_label, pred_label in zip(model_H, y_test, y_pred):
            true_label = int(true_label)
            pred_label = int(pred_label)

            # Solo considerar los casos donde la clase predicha es igual a la clase actual en el bucle
            if pred_label == pred_class:
                # Encontrar el rango de incertidumbre, limitando el índice a num_bins - 1 para evitar el error
                bin_index = min(np.digitize(H, uncertainty_bins) - 1, num_bins - 1)
                bin_label = f"{uncertainty_bins[bin_index]:.1f}-{uncertainty_bins[bin_index + 1]:.1f}"
                distribution[bin_label][true_label] += 1
                counts_per_bin[bin_label] += 1

        # Convertir a DataFrame y calcular el porcentaje de ocurrencias en cada rango de incertidumbre
        df = pd.DataFrame(distribution).T
        df = df.div(df.sum(axis=1), axis=0).fillna(0) * 100  # Convertir a porcentaje y llenar NaN con 0
        df["Total Images"] = counts_per_bin.values()  # Agregar columna con el total de imágenes por bin
        distributions_by_predicted_class[pred_class] = df

    # Guardar todas las distribuciones en un solo archivo CSV
    csv_file = os.path.join(output_dir, f"distributions_by_predicted_class.csv")

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')  # Usar ';' como separador
        writer.writerow(["Predicted Class", "Uncertainty Range", *range(num_classes), "Total Images"])  # Encabezados
        for pred_class, df in distributions_by_predicted_class.items():
            for bin_label, row in df.iterrows():
                writer.writerow([pred_class, bin_label, *row[:-1].values, row["Total Images"]])

    print(f"Distribuciones con incertirdumbre en: {csv_file}")
    return distributions_by_predicted_class




def collect_distribution_without_uncertanty(y_true, y_pred, output_dir="./Test", num_classes=6):
    """
    Genera una tabla de distribución de clases verdaderas para cada clase predicha,
    basada en las frecuencias observadas en el conjunto de entrenamiento.

    Parameters
    ----------
    y_true : ndarray
        Etiquetas verdaderas del conjunto de entrenamiento.
    y_pred : ndarray
        Etiquetas predichas por el modelo en el conjunto de entrenamiento.
    output_dir : str, opcional
        Directorio donde se guardará el archivo CSV.
    num_classes : int, opcional
        Número total de clases (default: 6).

    Returns
    -------
    DataFrame
        Un DataFrame que muestra la distribución de clases verdaderas
        para cada clase predicha.
    """

    # Inicializar una tabla de frecuencias para cada clase predicha
    distribution = {pred_class: {true_class: 0 for true_class in range(num_classes)} for pred_class in range(num_classes)}

    # Recopilar datos de frecuencias
    for true_label, pred_label in zip(y_true, y_pred):
        true_label = int(true_label)
        pred_label = int(pred_label)
        distribution[pred_label][true_label] += 1

    # Convertir a DataFrame y calcular frecuencias relativas
    df = pd.DataFrame(distribution).T  # Transponer para que las filas sean las clases predichas
    df = df.div(df.sum(axis=1), axis=0).fillna(0) * 100  # Convertir a porcentajes y manejar NaN
    df.columns = [f"True Class {i}" for i in range(num_classes)]
    df.index.name = "Predicted Class"

    # Guardar en un archivo CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_file = os.path.join(output_dir, "class_distribution_without_uncertainty.csv")
    df.to_csv(csv_file, sep=';')
    print(f"Distribución sin incertidumbre en: {csv_file}")

    return df




# ANALYSIS FUNCTIONS
# =============================================================================

def reliability_diagram(predictions, y_test, num_groups=10):
    """Generates the `reliability diagram` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    num_groups : int, optional (default: 10)
        Number of groups in which the prediction will be divided
        according to their predicted probability.
    
    Returns
    -------
    result : list of float
        List of the observed probabilities of each one of the predicted
        probability groups.
    """
    
    # Get number of classes
    num_classes = predictions.shape[2]
    
    # Calculate the bayesian samples average
    prediction = np.mean(predictions, axis=0)
    # print(prediction)
    
    # Labels to one-hot encoding
    labels = np.zeros((len(y_test), num_classes))
    labels[np.arange(len(y_test)), y_test] = 1

    # Probability groups to divide predictions
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    p_groups[-1] += 0.1 # To include the last value [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.1]
    
    result = []
    for i in range(num_groups):
        
        # Calculate the average of each group
        group = labels[(prediction >= p_groups[i]) &
                       (prediction < p_groups[i + 1])]
        if len(group) > 0:
            group_prob = group.sum() / len(group)
        else:
            group_prob = 0

        result.append(group_prob)

    return result # ejemplo: [0.05591177224929469, 0.2511261261261261, 0.23337091319052988, 0.24573170731707317, 0.23655913978494625, 0.5, 0, 0, 0, 0]

def accuracy_vs_uncertainty(predictions, y_test, H_limit=1.5, num_groups=10, analize_fpos=True):
    """Generates the `accuracy vs uncertainty` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    H_limit : float, optional (default: 1.5)
        The max value of the range of uncertainty.
    num_groups : int, optional (default: 15)
        Number of groups in which the prediction will be divided
        according to their uncertainty.
    
    Returns
    -------
    H_acc : list of float
        List of the accuracies of each one of the uncertainty groups.
    p_pixels : list of float
        List of the percentage of pixels belonging to each one of the
        uncertainty groups.
    """
    
    # Get predictive entropy
    test_H = _predictive_entropy(predictions)
    # print(f"Max predictive entropy: {test_H.max()}. Min predictive entropy: {test_H.min()}")

    # Generate a boolean map of hits
    y_pred = np.mean(predictions, axis=0).argmax(axis=1)
    test_ok = y_pred == y_test # [False  True False ... False False False]

    # Casos predichos como 0, pero reales son 1, 2, 3, 4 o 5
    false_positives = (y_pred == 1) & (y_test == 0)
    accuracy = np.mean(test_ok) * 100

    # Uncertainty groups to divide predictions
    H_groups = np.linspace(0.0, H_limit, num_groups + 1) # [0.   0.15 0.3  0.45 0.6  0.75 0.9  1.05 1.2  1.35 1.5 ]

    H_acc = []
    p_pixels = []
    for i in range(num_groups):
        
        # Calculate the average and percentage of pixels of each group
        group = test_ok[(test_H >= H_groups[i]) & (test_H < H_groups[i + 1])]
        p_pixels.append(len(group)/len(y_test))
        if len(group) > 0:
            group_prob = group.sum() / len(group)
        else:
            group_prob = 0

        H_acc.append(group_prob)

    # weighted_accuracy = sum([H_acc[i] * p_pixels[i] for i in range(num_groups)])
    # print(f"Weighted Global Accuracy from Groups: {weighted_accuracy * 100:.2f}%")
    return H_acc, p_pixels

def analyse_entropy(predictions, y_test):
    """Calculates the average uncertainty values by class
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    
    Returns
    -------
    class_H_avg : ndarray
        List of the averages of the global uncertainty (H) of each
        class. The last position also contains the average of the
        entire image.
    class_Ep_avg : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    class_H_Ep_avg : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    """
    
    # Get the uncertainty values
    model_H = _predictive_entropy(predictions)
    model_Ep = _expected_entropy(predictions)
    model_H_Ep = model_H - model_Ep
    
    # Structures for the averages
    num_classes = predictions.shape[2]
    class_H = np.zeros(num_classes + 1)
    class_Ep = np.zeros(num_classes + 1)
    class_H_Ep = np.zeros(num_classes + 1)
    class_px = np.zeros(num_classes + 1, dtype='int')
    
    for px, (H, Ep, H_Ep, label) in enumerate(zip(model_H, model_Ep,
                                                  model_H_Ep, y_test)):
        
        # Label as integer
        label = int(label)
        
        # Accumulate uncertainty values by class
        class_H[label] += H
        class_Ep[label] += Ep
        class_H_Ep[label] += H_Ep
        
        # Count pixels for class average
        class_px[label] += 1
        
        # Accumulate for every class
        class_H[-1] += H
        class_Ep[-1] += Ep
        class_H_Ep[-1] += H_Ep
        
        # Count pixels for global average
        class_px[-1] += 1

    # Return averages
    return class_H/class_px, class_Ep/class_px, class_H_Ep/class_px


def analyse_false_negative_entropy(predictions, y_test, y_pred, num_classes=6):
    """
    Calculates the average uncertainty values for false negatives where the model
    predicted class 0 but the true label was not 0.

    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Array of true labels.
    y_pred : ndarray
        Array of predicted labels by the model.
    num_classes : int
        Total number of classes (default: 6).

    Returns
    -------
    false_neg_class_H_avg : ndarray
        Average global uncertainty (H) per class, with the last element being the global average.
    false_neg_class_Ep_avg : ndarray
        Average aleatoric uncertainty (Ep) per class, with the last element being the global average.
    false_neg_class_H_Ep_avg : ndarray
        Average epistemic uncertainty (H - Ep) per class, with the last element being the global average.
    """

    # Get the uncertainty values
    model_H = _predictive_entropy(predictions)
    model_Ep = _expected_entropy(predictions)
    model_H_Ep = model_H - model_Ep

    # Initialize accumulators for uncertainty values
    class_H = np.zeros(num_classes + 1)
    class_Ep = np.zeros(num_classes + 1)
    class_H_Ep = np.zeros(num_classes + 1)
    class_px = np.zeros(num_classes + 1, dtype='int')

    # Loop through each sample
    for H, Ep, H_Ep, true_label, pred_label in zip(model_H, model_Ep, model_H_Ep, y_test, y_pred):

        # Check if it's a false negative: predicted 0 but true label is not 0
        if pred_label == 0 and true_label != 0:
            label = int(true_label)
            # Accumulate uncertainty values for each class
            class_H[label] += H
            class_Ep[label] += Ep
            class_H_Ep[label] += H_Ep
            class_px[label] += 1

            # Accumulate for overall average
            class_H[-1] += H
            class_Ep[-1] += Ep
            class_H_Ep[-1] += H_Ep
            class_px[-1] += 1

    # Avoid division by zero and calculate averages
    class_H_avg = np.divide(class_H, class_px, where=class_px != 0)
    class_Ep_avg = np.divide(class_Ep, class_px, where=class_px != 0)
    class_H_Ep_avg = np.divide(class_H_Ep, class_px, where=class_px != 0)

    return class_H_avg, class_Ep_avg, class_H_Ep_avg


def collect_uncertainty_by_case(predictions, y_test, y_pred, num_classes=6, uncertainty_type="predictive"):
    """
    Recopila los valores de incertidumbre para falsos negativos, falsos positivos y aciertos.

    Parameters
    ----------
    predictions : ndarray
        Array con las predicciones bayesianas.
    y_test : ndarray
        Etiquetas verdaderas del conjunto de prueba.
    y_pred : ndarray
        Etiquetas predichas por el modelo.
    num_classes : int, opcional
        Número total de clases (default: 6).
    uncertainty_type : str, opcional
        Tipo de incertidumbre que se recopilará ("predictive", "aleatoric" o "epistemic").

    Returns
    -------
    uncertainty_by_case : dict
        Diccionario con listas de incertidumbre para cada caso:
        - "false_negatives": Falsos negativos (predijo 0, etiqueta verdadera > 0).
        - "false_positives": Falsos positivos (predijo > 0, etiqueta verdadera 0).
        - "correct_predictions": Aciertos.
        Cada entrada contiene un diccionario con listas de incertidumbre por clase.
    """

    # Calcular las incertidumbres predictiva, aleatoria y epistémica
    model_H = _predictive_entropy(predictions)
    model_Ep = _expected_entropy(predictions)
    model_H_Ep = model_H - model_Ep

    # Elegir el tipo de incertidumbre
    if uncertainty_type == "predictive":
        uncertainty_values = model_H
    elif uncertainty_type == "aleatoric":
        uncertainty_values = model_Ep
    elif uncertainty_type == "epistemic":
        uncertainty_values = model_H_Ep
    else:
        raise ValueError("El tipo de incertidumbre debe ser 'predictive', 'aleatoric' o 'epistemic'")

    # Diccionario para almacenar incertidumbre de cada caso y clase
    uncertainty_by_case = {
        "false_negatives": {i: [] for i in range(1, num_classes)},  # Falsos negativos excluyen la clase 0
        "false_positives": {i: [] for i in range(1, num_classes)},  # Falsos positivos excluyen la clase 0
        "correct_predictions": {i: [] for i in range(num_classes)}  # Aciertos incluyen todas las clases
    }

    # Recopilar incertidumbre por caso
    for uncertainty, true_label, pred_label in zip(uncertainty_values, y_test, y_pred):
        true_label = int(true_label)
        pred_label = int(pred_label)

        # Falsos negativos: el modelo predice 0 y la clase real es diferente de 0
        if pred_label == 0 and true_label != 0:
            uncertainty_by_case["false_negatives"][true_label].append(uncertainty)

        # Falsos positivos: el modelo predice una clase mayor a 0, pero la etiqueta es 0
        elif pred_label != 0 and true_label == 0:
            uncertainty_by_case["false_positives"][pred_label].append(uncertainty)

        # Aciertos: predicción correcta
        elif pred_label == true_label:
            uncertainty_by_case["correct_predictions"][true_label].append(uncertainty)

    return uncertainty_by_case



def map_prediction(predictions):
    """Returns the bayesian predictions and global uncertainties (H)
    
    This function is implemented to facilitate all the data required
    for the maps comparisons.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    test_H : ndarray
        Array with the global uncertainty (H) values.
    """
    
    # Calculate the bayesian samples average prediction
    pred_map = np.mean(predictions, axis=0).argmax(axis=1)
    
    # Get the global uncertainty values
    test_H = _predictive_entropy(predictions)
    
    return pred_map, test_H

# PREDICTIONS FUNCTION
# =============================================================================

def bayesian_predictions(model, X_test, samples=100):
    """Generates bayesian predictions
    
    Parameters
    ----------
    model : TensorFlow Keras Sequential
        Trained bayesian model.
    X_test : ndarray
        Testing data set.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.
    
    Returns
    -------
    predictions : ndarray
        Array with the bayesian predictions.
    """
    
    # Bayesian stochastic passes
    predictions = []
    for i in range(samples):
        
        # Progress bar
        status = int(78*len(predictions)/samples)
        print('[' + '='*(status) + ' '*(78 - status) + ']', end="\r",
              flush=True)
        
        # Launch prediction
        prediction = model.predict(X_test, verbose=0)
        # print(f"Prediction: {prediction}: {prediction.shape}")
        summed_predictions = prediction.sum(axis=1)
        # print("Shape summed predictions: ", summed_predictions.shape)
        rounded_predictions = np.round(summed_predictions).astype(int)
        # print("Shape rounded_predictions: ", rounded_predictions.shape)
        num_classes = 6  # Define el número máximo de categorías (0 a 5)
        one_hot_predictions = np.eye(num_classes)[rounded_predictions]

        # Imprimir la forma y el contenido de la predicción sumada
        # print("Shape of summed_predictions:", one_hot_predictions.shape)
        # print("Summed predictions:", one_hot_predictions)
        class_counts = one_hot_predictions.sum(axis=0)

        # Imprimir el conteo de cada clase
        # for i, count in enumerate(class_counts):
        #    print(f"Class {i}: {int(count)}")
        predictions.append(one_hot_predictions)

    
    # End of progress bar
    print('[' + '='*78 + ']', flush=True)
    
    return np.array(predictions)


def smooth_one_hot_predictions(summed_predictions, num_classes):
    # Inicializar una matriz de predicciones suaves con ceros
    smooth_predictions = torch.zeros((summed_predictions.size(0), num_classes))

    # Iterar a través de cada predicción y hacer la interpolación suave
    for i, value in enumerate(summed_predictions):
        # Obtener el índice inferior y superior de las clases más cercanas
        lower_class = int(torch.floor(value))
        upper_class = lower_class + 1

        # Calcular la distancia del valor a la clase inferior
        upper_weight = value - lower_class
        lower_weight = 1 - upper_weight  # El complemento será el peso para la clase inferior

        # Asignar los pesos correspondientes a cada clase
        if lower_class < num_classes:
            smooth_predictions[i, lower_class] = lower_weight
        if upper_class < num_classes:
            smooth_predictions[i, upper_class] = upper_weight

    return smooth_predictions



def bayesian_predictions_torch(model, X_test, samples=100):
    """Generates bayesian predictions

    Parameters
    ----------
    model : TensorFlow Keras Sequential
        Trained bayesian model.
    X_test : ndarray
        Testing data set.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.

    Returns
    -------
    predictions : ndarray
        Array with the bayesian predictions.
    """

    # Bayesian stochastic passes
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(samples):
            # Progreso bar
            status = int(78 * i / samples)
            print('[' + '=' * status + ' ' * (78 - status) + ']', end="\r", flush=True)

            # Realizar una pasada estocástica (inferencia)
            prediction = model(X_test)
            prediction = torch.sigmoid(prediction) # .round()
            summed_predictions = prediction.sum(dim=1) # .long()
            num_classes = 6
            one_hot_predictions = smooth_one_hot_predictions(summed_predictions, num_classes)
            # one_hot_predictions = torch.eye(num_classes)[summed_predictions.long()]
            predictions.append(one_hot_predictions.cpu().numpy())
    predictions = np.array(predictions)
    # End of progress bar
    print('[' + '=' * 78 + ']', flush=True)

    return predictions



def bayesian_predictions_torch_2(model, X_test, samples=100):
    """Generates bayesian predictions

    Parameters
    ----------
    model : TensorFlow Keras Sequential
        Trained bayesian model.
    X_test : ndarray
        Testing data set.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.

    Returns
    -------
    predictions : ndarray
        Array with the bayesian predictions.
    """

    # Bayesian stochastic passes
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(samples):
            # Progreso bar
            status = int(78 * i / samples)
            print('[' + '=' * status + ' ' * (78 - status) + ']', end="\r", flush=True)

            # Realizar una pasada estocástica (inferencia)
            prediction = model(X_test)
            predictions.append(prediction.cpu().numpy())
    predictions = np.array(predictions)
    # End of progress bar
    print('[' + '=' * 78 + ']', flush=True)

    return predictions


