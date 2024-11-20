import numpy as np
import torch
import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd
import os


def calculate_accuracy_from_df(pred_col, real_col):
    """
    Calcula el accuracy de las predicciones en un DataFrame.

    Args:
    df (pd.DataFrame): DataFrame que contiene las predicciones y las etiquetas reales.
    pred_col (str): Nombre de la columna que contiene las predicciones.
    real_col (str): Nombre de la columna que contiene las etiquetas reales (ground truth).

    Returns:
    float: Accuracy en porcentaje.
    """
    # Comparar las predicciones con las etiquetas reales
    correct = (pred_col == real_col).sum()

    # Calcular el accuracy
    accuracy = correct / len(df)

    return accuracy * 100


data_dir = '/data'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv')) # train  train_prueba
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))


model_dir = '../input/panda-public-models'
image_folder = os.path.join(data_dir, 'test_images')
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
image_folder = image_folder if is_test else os.path.join(data_dir, 'train_images') # prueba_images  train_images

df = df_test if is_test else df_train


# Cargar las activaciones intermedias
feature_activations = np.load("/data/hook/feature_activations.npy")
feature_activations = torch.tensor(feature_activations, dtype=torch.float32)
weights_path = "/data/hook/myfc_weights.npy"
bias_path = "/data/hook/myfc_bias.npy"
myfc_weights = np.load(weights_path)
myfc_bias = np.load(bias_path)
myfc_weights_tensor = torch.tensor(myfc_weights, dtype=torch.float32)
myfc_bias_tensor = torch.tensor(myfc_bias, dtype=torch.float32)

input_dim = myfc_weights.shape[1]
output_dim = myfc_weights.shape[0]
myfc = nn.Linear(input_dim, output_dim)

# Cargar los pesos y sesgos en la nueva capa
myfc.weight = nn.Parameter(myfc_weights_tensor)
myfc.bias = nn.Parameter(myfc_bias_tensor)

print("Capa myfc creada con los pesos y sesgos cargados")


myfc.eval()

# Pasar las activaciones a través de la capa final `myfc`
with torch.no_grad():
    predictions = myfc(feature_activations)

# Aplicar activación sigmoide si es necesario (según el modelo original)
predictions = torch.sigmoid(predictions)
predictions_sum = predictions.sum(dim=2)  # Resultado: (10616, 1)

# Redondear para obtener valores enteros
predictions_final = predictions_sum.round().squeeze()  # Elimina dimensiones adicionales

# Convertir a numpy para el análisis o guardado
predictions_final = predictions_final.numpy()

# Ejemplo: guardar las predicciones finales
np.save("/data/hook/predictions_from_activations.npy", predictions)
print("Predicciones finales guardadas en predictions_from_activations.npy")
print(predictions_final.shape)
print(predictions_final)

real_col = df['isup_grade']
df['isup_grade'] = predictions_final

accuracy = calculate_accuracy_from_df(df['isup_grade'], real_col)
print(f"Accuracy: {accuracy}%")