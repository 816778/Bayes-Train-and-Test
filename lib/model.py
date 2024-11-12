#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model module of the bnn4hi package

This module defines the bayesian model used to train.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow_probability import distributions as dist
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Add, Conv2D, MaxPooling2D, Dense

import tensorflow as tf
import tensorflow_probability as tfp


# Definir la capa bayesiana que va a reemplazar la capa myfc del modelo anterior
class BayesianModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(BayesianModel, self).__init__()
        self.prior_fn = tfp.layers.default_multivariate_normal_fn
        self.posterior_fn = tfp.layers.default_mean_field_normal_fn(is_singular=False)

        # Capa bayesiana en lugar de la capa `myfc`
        self.bayesian_fc = tfp.layers.DenseVariational(
            units=output_dim,
            make_posterior_fn=self.posterior_fn,
            make_prior_fn=self.prior_fn,
            kl_weight=1 / input_dim[-1],  # peso del término KL para el entrenamiento
        )

    def call(self, inputs):
        # Pasamos las características extraídas por la capa bayesiana
        logits = self.bayesian_fc(inputs)
        return logits




# MODEL FUNCTION
# Define the Bayesian Neural Network
tfd = tfp.distributions
tfpl = tfp.layers

# Definimos la pérdida personalizada (Mean Squared Error)
def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Definimos la métrica personalizada
def custom_accuracy(y_true, y_pred):
    y_pred_rounded = tf.math.round(y_pred)  # Redondeamos las predicciones
    correct_predictions = tf.reduce_all(tf.equal(y_true, y_pred_rounded), axis=1)  # Comparación binaria
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))  # Calculamos precisión promedio
    return accuracy


# Definir la red bayesiana
def create_bayesian_model(input_shape, num_classes, learning_rate, shape, modelo=0, type="fc"):
    tf.keras.backend.clear_session()
    if "fc" in type:
        if modelo == 0:
            """
            EPOCH 
            ACCURACY: 
            """
            pretrained_weights = np.load('/data/hook/myfc_weights.npy')
            pretrained_bias = np.load('/data/hook/myfc_bias.npy')
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tfpl.DenseFlipout(num_classes, activation='sigmoid')  # Salida entre 0 y 1
            ])

            dense_flipout_layer = model.layers[0]

            # Cargar los pesos preentrenados como las medias
            # pretrained_weights.shape = (1280, 5), pretrained_bias.shape = (5,)
            pretrained_weights_mean = pretrained_weights.T  # Transponer para tener forma (1280, 5)
            pretrained_bias_mean = pretrained_bias

            # Inicializamos con desviación estándar pequeña para los pesos
            pretrained_weights_stddev = np.full(pretrained_weights_mean.shape, 0.01)
            pretrained_bias_stddev = np.full(pretrained_bias_mean.shape, 0.01)

            # Ajustar los pesos en la capa DenseFlipout
            dense_flipout_layer.set_weights([pretrained_weights_mean, pretrained_weights_stddev, pretrained_bias_mean])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif modelo == 1:
            model = tf.keras.Sequential([
                tfp.layers.DenseFlipout(units=num_classes, input_shape=input_shape),
                tf.keras.layers.Activation('sigmoid')  # Sigmoid si necesitas una salida entre 0 y 1
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])
        else:
            print("MODELO = 2")
            model = None
    else:
        print("MODELO = NO FC")
        model = None
    # Resumen del modelo
    model.summary()
    return model
