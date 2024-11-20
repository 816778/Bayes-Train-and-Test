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
import torchbnn as bnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Local imports
if '.' in __name__:

    # To run as a module
    from .lib.model import create_bayesian_model
    from .lib.bayesian_model import BayesianENet
    from .lib import my_config
    from .lib.data import get_dataset, get_mixed_dataset, get_filtered_dataset
    # from .lib.BayesianNN import BayesianNN

else:
    # To run as an script
    from lib.model import create_bayesian_model
    from lib.bayesian_model import BayesianENet
    from lib import my_config
    from lib.data import get_dataset, get_mixed_dataset, get_filtered_dataset
    # from lib.BayesianNN import BayesianNN


# PARAMETERS
# =============================================================================
def _parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("data_path", help="Ruta a la carpeta con los archivos CSV de datos")
    parser.add_argument("csv_path", help="Ruta a la carpeta con los archivos CSV de datos")
    parser.add_argument("epochs", type=int, help="Total number of epochs to train.")
    parser.add_argument("dataset_name", help="Nombre del conjunto de datos para la generación del archivo de salida")
    parser.add_argument("modelo", type=int, help="Número de modelo que se va a ejecutar")
    parser.add_argument('--l1_n', type=int, default=128, help="Número de neuronas en la primera capa oculta")
    parser.add_argument('--l2_n', type=int, default=64, help="Número de neuronas en la segunda capa oculta")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Tasa de aprendizaje")

    return parser.parse_args()


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

def save_model(model, output_dir, epoch):
    model_path = os.path.join(output_dir, f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

def step_encode_labels(y, num_classes):
    label = np.zeros(num_classes, dtype=np.float32)  # Inicializa con ceros
    label[:y] = 1.0  # Asigna 1 a los primeros y elementos
    return label

# Transformamos y_train y y_val
def transform_labels(y_data, num_classes):
    return np.array([step_encode_labels(y, num_classes) for y in y_data])



# MAIN FUNCTION
# =============================================================================
def train(layer_name, epochs, modelo=1):
    """Trains a bayesian model for a hyperspectral image dataset

        The trained model and the checkouts are saved in the `MODELS_DIR`
        defined in `config.py`.

        Parameters
        ----------

        """

    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
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
    X_train, y_train, X_val, y_val, _, _ = get_dataset(file_path=args.data_path, csv_path=args.csv_path, seed=35)
    # X_train, y_train, X_val, y_val, _, _ = get_filtered_dataset(args.data_path, args.csv_path, 
    #    csv_file_uncertanty='./Test/image_uncertainties_predictive.csv', threshold=0.6, seed=35,)


    X_train_tensor = torch.tensor(X_train.squeeze(1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.squeeze(1), dtype=torch.float32)

    # Transformar y_train y y_val a "step encoding"
    y_train_encoded = transform_labels(y_train, num_classes)
    y_val_encoded = transform_labels(y_val, num_classes)

    # Convertimos a tensores de PyTorch
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.float32)

    print(f'Size Training set: {X_train_tensor.shape} - {y_train_tensor.shape}')
    print(f'Size Validation set: {X_train_tensor.shape} - {y_val_tensor.shape}\n')

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    output_dir = (f"{layer_name}_{l1_n}-{l2_n}model_{p_train}train"
                  f"_{learning_rate}lr")
    output_dir = os.path.join(my_config.MODELS_DIR, output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    input_shape = X_train.shape[2:]


    ####################################################
    print(f"Modelo: {modelo}")
    model = BayesianENet(modelo=int(modelo), in_features=input_shape[0], output_dim=num_classes)
    optimizer, criterion, _ = model.setup_optimizer_and_criterion(learning_rate)
    model.summary(input_shape)
    ####################################################

    for epoch in range(epochs):
        model.train()  # Configura el modelo en modo de entrenamiento
        running_loss = 0.0
        running_corrects = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

            # Calcular el accuracy
            y_pred_rounded = torch.sigmoid(y_pred).round()
            y_pred_sum = y_pred_rounded.sum(dim=1)
            y_batch_sum = y_batch.sum(dim=1)

            running_corrects += (y_pred_sum == y_batch_sum).float().sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = (running_corrects / len(train_loader.dataset)) * 100


        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():  # Desactivar el cálculo de gradientes
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                loss = criterion(y_val_pred, y_val_batch)
                val_loss += loss.item() * X_val_batch.size(0)

                y_val_pred_rounded = torch.sigmoid(y_val_pred).round()
                y_val_pred_sum = y_val_pred_rounded.sum(dim=1)
                y_val_batch_sum = y_val_batch.sum(dim=1)
                val_corrects += (y_val_pred_sum == y_val_batch_sum).float().sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = (val_corrects / len(val_loader.dataset)) * 100

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if epoch + 1 == epochs:
            save_model(model, output_dir, epoch + 1)

    exit()


if __name__ == "__main__":
    args = _parse_args()
    train(args.dataset_name, args.epochs, args.modelo)
