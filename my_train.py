import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import absl.logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import  warnings
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Local imports
if '.' in __name__:

    # To run as a module
    from .lib.model import create_bayesian_model
    from .lib.bayesian_model import BayesianENet
    from .lib import my_config
    from .lib.data import get_dataset, get_mixed_dataset
    # from .lib.BayesianNN import BayesianNN

else:
    # To run as an script
    from lib.model import create_bayesian_model
    from lib.bayesian_model import BayesianENet
    from lib import my_config
    from lib.data import get_dataset, get_mixed_dataset
    # from lib.BayesianNN import BayesianNN


# PARAMETERS
# =============================================================================
def _parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("data_path", help="Ruta a la carpeta con los archivos CSV de datos")
    parser.add_argument("csv_path", help="Ruta a la carpeta con los archivos CSV de datos")
    parser.add_argument("epochs", type=int, help="Total number of epochs to train.")
    parser.add_argument("period", type=int, help="Checkpoints and information period.")
    parser.add_argument("output_dir", help="Directorio donde se guardarán los resultados")
    parser.add_argument("dataset_name", help="Nombre del conjunto de datos para la generación del archivo de salida")
    # parser.add_argument("train_file_path", help="Nombre del fichero dataset train")
    # parser.add_argument("val_file_path", help="Nombre del fichero dataset validacion")
    parser.add_argument("modelo", type=int, default=0, help="Número de modelo que se va a ejecutar")
    parser.add_argument('--l1_n', type=int, default=128, help="Número de neuronas en la primera capa oculta")
    parser.add_argument('--l2_n', type=int, default=64, help="Número de neuronas en la segunda capa oculta")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Tasa de aprendizaje")
    # train_file_path, val_file_path

    return parser.parse_args()


class _PrintCallback(tf.keras.callbacks.Callback):
    """Callback to print time, loss and accuracy logs during training

    Callbacks can be passed to keras methods such as `fit`, `evaluate`,
    and `predict` in order to hook into the various stages of the model
    training and inference lifecycle.

    Attributes
    ----------
    print_epoch : int
        The log messages are written each `print_epoch` epochs.
    losses_avg_no : int
        The current loss value is calculated as the average of the last
        `losses_avg_no` batches loss values.
    start_epoch : int
        Number of the initial epoch in case of finetuning.

    Methods
    -------
    print_loss_acc(self, logs, time, last=False)
        Prints log messages with time, loss and accuracy values.
    on_train_begin(self, logs={})
        Called at the beginning of training. Instantiates and
        initialises the `losses`, `epoch` and `start_time` attributes.
    on_batch_end(self, batch, logs={})
        Called at the end of a training batch in `fit` methods.
        Actualises the `losses` attribute with the current value of the
        `loss` item in `logs` dict.
    on_epoch_end(self, epoch, logs={})
        Called at the end of an epoch. Actualises epoch counter and
        prints log message on printable epochs.
    on_train_end(self, logs={})
        Called at the end of training. Prints end of training log
        message.
    """

    def __init__(self, print_epoch=1000, losses_avg_no=100, start_epoch=0):
        """Inits PrintCallback instance

        Parameters
        ----------
        print_epoch : int, optional (default: 1000)
            The log messages are written each `print_epoch` epochs.
        losses_avg_no : int, optional (default: 100)
            The current loss value is calculated as the average of the
            last `losses_avg_no` batches loss values.
        start_epoch : int, optional (default: 0)
            Number of the initial epoch in case of finetuning.
        """
        self.print_epoch = print_epoch
        self.losses_avg_no = losses_avg_no
        self.start_epoch = start_epoch

    def print_loss_acc(self, logs, time, last=False):
        """Prints log messages with time, loss and accuracy values

        Parameters
        ----------
        logs : dict
            Aggregated metric results up until this batch.
        time : int
            Current training time in seconds.
        last : bool, optional (default: False)
            Flag to activate end of training log message.
        """

        # Calculate current loss value
        loss = sum(self.losses[-self.losses_avg_no:]) / self.losses_avg_no

        # Print log message
        if last:
            print(f"\n--- TRAIN END AT EPOCH {self.epoch} ---")
            print(f"TRAINING TIME: {time} seconds")
            end = "\n"
        else:
            print(f"\nCURRENT TIME: {time} seconds")
            end = ''
        print(f"Epoch loss ({self.epoch}): {loss}")
        print(f"Accuracy: {logs.get('val_accuracy')}", end=end, flush=True)

        def on_train_begin(self, logs={}):
            """Called at the beginning of training

            Instantiates and initialises the `losses`, `epoch` and
            `start_time` attributes. The `logs` parameter is not used, but
            this is an overwritten method, so it is mandatory.

            Parameters
            ----------
            logs : dict
                Currently no data is passed to this argument for this
                method but that may change in the future.
            """
            self.losses = []
            self.epoch = self.start_epoch
            self.start_time = time.time()

        def on_batch_end(self, batch, logs={}):
            """Called at the end of a training batch in `fit` methods

            Actualises the `losses` attribute with the current value of the
            `loss` item in `logs` dict. The `batch` parameter is not used,
            but this is an overwritten method, so it is mandatory.

            This is a backwards compatibility alias for the current method
            `on_train_batch_end`.

            Note that if the `steps_per_execution` argument to `compile` in
            `tf.keras.Model` is set to `N`, this method will only be called
            every `N` batches.

            Parameters
            ----------
            batch : int
                Index of batch within the current epoch.
            logs : dict
                Aggregated metric results up until this batch.
            """
            self.losses.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            """Called at the end of an epoch

            Actualises epoch counter and prints log message on printable
            epochs.

            This function should only be called during TRAIN mode.

            Parameters
            ----------
            epoch : int
                Index of epoch.
            logs : dict
                Metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation
                result keys are prefixed with `val_`. For training epoch,
                the values of the `Model`'s metrics are returned.
            """

            # Actualise epoch
            self.epoch += 1

            # If it is a printable epoch
            if self.epoch % self.print_epoch == 0:
                # Print log message
                current_time = time.time() - self.start_time
                self.print_loss_acc(logs, current_time)

        def on_train_end(self, logs={}):
            """Called at the end of training

            Prints end of training log message.

            Parameters
            ----------
            logs : dict
                Currently the output of the last call to `on_epoch_end()`
                is passed to this argument for this method but that may
                change in the future.
            """
            total_time = time.time() - self.start_time
            self.print_loss_acc(logs, total_time, last=True)


def intercambiar_etiquetas(y, clase1, clase2, random_state=None):
    """
    Intercambia la mitad de las etiquetas entre dos clases en un array de etiquetas.

    Parameters
    ----------
    y : ndarray
        Array de etiquetas.
    clase1 : int
        Primera clase para el intercambio.
    clase2 : int
        Segunda clase para el intercambio.
    random_state : int, optional
        Semilla para la aleatorización, por defecto None.

    Returns
    -------
    y_modificado : ndarray
        Array de etiquetas modificado después del intercambio.
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Encontrar las posiciones de las etiquetas de clase1 y clase2
    idx_clase1 = np.where(y == clase1)[0]
    idx_clase2 = np.where(y == clase2)[0]

    # Determinar el número de etiquetas a intercambiar (mitad del menor)
    n_intercambiar = min(len(idx_clase1), len(idx_clase2)) // 2

    # Seleccionar aleatoriamente la mitad de las etiquetas de cada clase
    idx_intercambio_clase1 = np.random.choice(idx_clase1, n_intercambiar, replace=False)
    idx_intercambio_clase2 = np.random.choice(idx_clase2, n_intercambiar, replace=False)

    # Intercambiar las etiquetas
    y_modificado = y.copy()
    y_modificado[idx_intercambio_clase1] = clase2
    y_modificado[idx_intercambio_clase2] = clase1

    return y_modificado

def step_encode_labels(y, num_classes):
    label = np.zeros(num_classes, dtype=np.float32)  # Inicializa con ceros
    label[:y] = 1.0  # Asigna 1 a los primeros y elementos
    return label

# Transformamos y_train y y_val
def transform_labels(y_data, num_classes):
    return np.array([step_encode_labels(y, num_classes) for y in y_data])


    # MAIN FUNCTION
    # =============================================================================
def train(layer_name, period, epochs, modelo=0):
    """Trains a bayesian model for a hyperspectral image dataset

        The trained model and the checkouts are saved in the `MODELS_DIR`
        defined in `config.py`.

        Parameters
        ----------

        """

    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    # Input, output and dataset references
    d_path = my_config.DATA_PATH
    base_output_dir = my_config.MODELS_DIR

    # Model parameters
    l1_n = my_config.LAYER1_NEURONS
    l2_n = my_config.LAYER2_NEURONS

    # Training parameters
    p_train = my_config.P_TRAIN
    learning_rate = my_config.LEARNING_RATE

    num_classes = my_config.NUM_CLASES_TRAIN

    # GET DATA
    # ---------------------------------------------------------------------
    # Get dataset
    X_train, y_train, X_val, y_val, _, _ = get_dataset(args.data_path, args.csv_path, 6)
    X_train = np.squeeze(X_train)
    X_val = np.squeeze(X_val)

    print(f'Tamaño del conjunto de entrenamiento: {len(X_train)}')
    print(f'Tamaño del conjunto de validación: {len(X_val)}')
    if isinstance(X_train, torch.Tensor) or isinstance(X_train, np.ndarray):
        print("X_train shape:", X_train.shape)
    print("X_train type:", type(X_train))

    output_dir = (f"{layer_name}_{l1_n}-{l2_n}model_{p_train}train"
                  f"_{learning_rate}lr")
    output_dir = os.path.join(my_config.MODELS_DIR, output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    num_features = X_train.shape[1]
    dataset_size = len(X_train)
    input_shape = X_train.shape[1:]
    # input_shape = tuple(dim for dim in input_shape if dim != 1)

    print(f'num_features: {num_features}\nnum_classes: {num_classes}\ndataset_size: {dataset_size}\n')
    print(f'input_shape: {input_shape}\n')
    ####################################################
    model = create_bayesian_model(input_shape, num_classes, learning_rate, X_train.shape, modelo=modelo)
    ####################################################
    y_train_encoded = transform_labels(y_train, num_classes)
    y_val_encoded = transform_labels(y_val, num_classes)
    if isinstance(y_train_encoded, torch.Tensor) or isinstance(y_train_encoded, np.ndarray):
        print("y_train_encoded shape:", y_train_encoded.shape)
    print("y_train_encoded type:", type(y_train_encoded))

    print_callback = _PrintCallback(print_epoch=period, losses_avg_no=max(1, period // 10), start_epoch=0)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, "epoch_{epoch}"),
                                                    monitor='val_accuracy', verbose=1, mode='max',
                                                    save_best_only=False, save_freq='epoch')

    print('#' * 80)
    print(f"\nOUTPUT DIR: {output_dir}", flush=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    model.fit(X_train, y_train_encoded, epochs=epochs, verbose=0,
              callbacks=[early_stopping, reduce_lr, print_callback, checkpoint],
              validation_data=(X_val, y_val_encoded))
    model.save(os.path.join(output_dir, "final"))
    print(f"Entrenamiento finalizado.\nModelo guardado en {output_dir}")
    exit()


if __name__ == "__main__":
    args = _parse_args()
    train(args.dataset_name, args.period, args.epochs, args.modelo)
