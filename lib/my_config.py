#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Config module of the bnn4hi package

This module provides macros with data information, training and testing
parameters and plotting configurations.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import numpy as np

# CONFIGURATION GLOBALS
# =============================================================================

# Uncomment if there are GPU memory errors during training
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Uncomment for CPU execution. Only recommended if there are GPU memory
# errors during testing
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Input and output directories
DATA_PATH = "./Data"
MODELS_DIR = "./Models"
TEST_DIR = "./Test"

# DATA INFORMATION
# =============================================================================

# Datasets
#     The classes to mix when training with mixed classes have been
#     selected for having enough and similar number of labelled pixels.
#     Wavelengths have been selected according to the sensor and the
#     characteristics of each image for a better RGB representation.
url_base = "http://www.ehu.es/ccwintco/uploads"

DATASETS_LIST = ["fc"]
DATASETS = {
    "fc": {
        'num_classes': 6,
        'num_features': 1280,
        'mixed_class_A': 4,
        'mixed_class_B': 5
    }
}


# TRAINING AND TESTING PARAMETERS
# =============================================================================

# Model parameters
LAYER1_NEURONS = 32
LAYER2_NEURONS = 16

# Training parameters
P_TRAIN = 0.5
LEARNING_RATE = 1.0e-2

# Bayesian passes
BAYESIAN_PASSES = 100

# List of noises for noise tests
NOISES = np.arange(0.0, 0.61, 0.3)

NUM_CLASES_TRAIN = 5

# PLOTTING CONFIGURATIONS
# =============================================================================
# Plots size
PLOT_W = 7
PLOT_H = 4

# Plots colours
COLOURS = {"fc": "#2B4162"}
          # "blocks_13": "#FA9F42",
          # "blocks_14": "#0B6E4F"}
