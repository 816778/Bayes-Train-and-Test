import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def transform_labels(y, num_classes=5):
    y_transformed = torch.zeros((y.size(0), num_classes), dtype=torch.float32)
    for i in range(y.size(0)):
        y_transformed[i, :int(y[i])] = 1
    return y_transformed

feature_activations = np.load("/data/hook/feature_activations.npy")
feature_activations = torch.tensor(feature_activations, dtype=torch.float32)  # Convertir a tensor de PyTorch

data_dir = '/data'
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
labels = torch.tensor(df['isup_grade'].values, dtype=torch.long)

labels_transformed = transform_labels(labels, num_classes=5)

X_train, X_val, y_train, y_val = train_test_split(feature_activations, labels_transformed, test_size=0.2, random_state=42)

input_dim = feature_activations.shape[-1]
output_dim = 5
print("input_dim: ", input_dim)
myfc = nn.Linear(input_dim, output_dim)

# Configurar optimizador y función de pérdida
optimizer = optim.Adam(myfc.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

n_epochs = 15
myfc.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Forward pass
    predictions = myfc(X_train).squeeze(1)  # Salida de tamaño (batch_size, 5)
    loss = criterion(predictions, y_train)

    # Backward pass y optimización
    loss.backward()
    optimizer.step()

    # Calcular accuracy aproximado (redondeando predicciones)
    predictions_rounded = torch.sigmoid(predictions).round()
    predictions_sum = predictions_rounded.sum(dim=1)
    y_train_sum = y_train.sum(dim=1)
    train_accuracy = (predictions_sum == y_train_sum).float().mean() * 100
    # train_accuracy = (predictions_rounded == y_train).float().mean() * 100
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy.item():.2f}%")

# Validación
myfc.eval()
with torch.no_grad():
    val_predictions = myfc(X_val).squeeze(1)
    val_loss = criterion(val_predictions, y_val)
    val_predictions_rounded = torch.sigmoid(val_predictions).round()
    print(val_predictions_rounded)
    print(y_val)
    val_predictions_sum = val_predictions_rounded.sum(dim=1)
    y_val_sum = y_val.sum(dim=1)
    val_accuracy = (val_predictions_sum == y_val_sum).float().mean() * 100
    # val_accuracy = (val_predictions_rounded == y_val).float().mean() * 100
    print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy.item():.2f}%")

save_dir = "/home/eryka/Bayes/Models/fc_32-16model_0.5train_0.01lr/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
torch.save(myfc.state_dict(), save_dir + f"epoch_{n_epochs}.pth")
print("Modelo myfc entrenado y guardado en " + save_dir)

