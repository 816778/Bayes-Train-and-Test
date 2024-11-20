import os
import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path
sys.path = [
    '../input/ttach-kaggle/ttach/',
] + sys.path

import argparse

# Manejo de argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Selecciona el modelo a usar")
parser.add_argument('--model_idx', type=int, choices=[0, 1, 2, 3, 4], default=0,
                    help="Índice del modelo a utilizar (0, 1, 2, 3 o 4)")

args = parser.parse_args()
import ttach as tta
import skimage.io
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet

import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = '/data'
df_train = pd.read_csv(os.path.join(data_dir, 'train_prueba.csv')) # train  train_prueba
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))


model_dir = '../input/panda-public-models'
image_folder = os.path.join(data_dir, 'test_images')
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.
image_folder = image_folder if is_test else os.path.join(data_dir, 'train_images') # prueba_images  train_images

df = df_test if is_test else df_train

tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 1
num_workers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(image_folder)


class enetv3(nn.Module):
    def __init__(self, modelname, out_dim=5, freeze_bn=True):
        super(enetv3, self).__init__()

        from efficientnet_pytorch import EfficientNet
        self.basemodel = EfficientNet.from_name(modelname)
        self.myfc = nn.Linear(self.basemodel._fc.in_features, out_dim)
        self.basemodel._fc = nn.Identity()
        self.basemodel._avg_pooling = nn.AdaptiveAvgPool2d(1)  # GeM()


        self.feature_activations = None
    def extract(self, x):
        return self.basemodel(x)

    def forward(self, x):
        x = self.basemodel(x)
        self.feature_activations = x.clone().detach().cpu().numpy()
        x = self.myfc(x)
        return x


def load_models(model_files):
    models = []
    for model_f in model_files:
        model_f = os.path.join("../input/latesubspanda", model_f)
        if "eff" in model_f:
            model = enetv3(modelname, out_dim=5)
        model.load_state_dict(torch.load(model_f))
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models


modelname="efficientnet-b0"

# load ensembles

model_files2 = [
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold0.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold1.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold2.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold3.pth",
    "efficientnet-b0famlabelsmodelsub_avgpool_tile36_imsize256_mixup_final_epoch20_fold4.pth"
]

selected_model_file = model_files2[args.model_idx]
print(f"Usando el modelo {selected_model_file}")
# models2 = load_models(model_files2)

# Cargamos solo el modelo seleccionado
models2 = load_models([selected_model_file])


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


def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
                  constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img) < n_tiles:
        img3 = np.pad(img3, [[0, N - len(img3)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img': img3[i], 'idx': i})
    return result, n_tiles_with_info >= n_tiles


class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 sub_imgs=False
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.sub_imgs = sub_imgs

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id

        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        image = skimage.io.ImageCollection(tiff_file)[1][:, :, ::-1]
        tiles, OK = get_tiles(image, self.tile_mode)

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))
        idxes = np.asarray(idxes) + self.n_tiles if self.sub_imgs else idxes

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w

                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1 + image_size, w1:w1 + image_size] = this_img
            # images = 255 - images
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        return torch.tensor(images)


"""
if not is_test:
    dataset_show = PANDADataset(df, image_size, n_tiles, 4)
    from pylab import rcParams
    rcParams['figure.figsize'] = 20,10
    for i in range(2):
        f, axarr = plt.subplots(1,5)
        for p in range(5):
            idx = np.random.randint(0, len(dataset_show))
            img = dataset_show[idx]
            axarr[p].imshow(1. - img.transpose(0, 1).transpose(1,2).squeeze())
            axarr[p].set_title(str(idx))
"""
print(df)
print("######################")
# Prepare tiles with different paddings, introducing slightly different tiles
dataset = PANDADataset(df, image_size, n_tiles, 0)  # mode == 0
loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

dataset2 = PANDADataset(df, image_size, n_tiles, 2)  # mode == 2
loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)

import ttach as tta

if len(dataset) > 3:
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180]),
        ]
    )
else:
    # For commits
    transforms = tta.Compose(
        [
            # tta.HorizontalFlip(),
        ]
    )

tta_models2 = []
# TTA wrappers
for model in models2:
    tta_models2.append(tta.ClassificationTTAWrapper(model, transforms))

LOGITS = []
LOGITS2 = []
LOGITS12 = []
LOGITS22 = []

num_models = len(tta_models2)
print(f"Tamaño Dataset: {len(dataset)}")
print(f"Num modelos: {num_models}")
imprimir = True
feature_activations_list = []
with torch.no_grad():
    for data in tqdm(loader):
        data = data.to(device)
        logits = None
        for i, tta_model in enumerate(tta_models2):
            if i == 0:
                logits = tta_model(data).sigmoid()
                if imprimir:
                    imprimir = False
                    print(f"Logits 1_1:\n{logits}")
                    # tensor([[3.8615e-02, 2.6129e-03, 7.2283e-04, 2.5589e-04, 7.3373e-05]], device='cuda:0')
            else:
                logits = logits + tta_model(data).sigmoid()
            feature_activations_list.append(tta_model.model.feature_activations)
        LOGITS2.append(logits / num_models)

    save_1 = np.array(feature_activations_list)

    imprimir = True
    feature_activations_list = []
    
    for data in tqdm(loader2):
        data = data.to(device)
        for i, tta_model in enumerate(tta_models2):
            if i == 0:
                logits = tta_model(data).sigmoid()
                if imprimir:
                    imprimir = False
                    print(f"Logits 1_2:\n{logits}")
            else:
                logits = logits + tta_model(data).sigmoid()
            feature_activations_list.append(tta_model.model.feature_activations)
            LOGITS22.append(logits / num_models)

    save_2 = np.array(feature_activations_list)
    save = (save_1 + save_2) / 2

    feature_activations_array = np.array(save)
    np.save("/data/hook/feature_activations_all.npy", feature_activations_array)
    print("Activaciones guardadas en feature_activations_all.npy")


LOGITS = (torch.cat(LOGITS2).cpu() + torch.cat(LOGITS22).cpu()) / 2
print(f"Shape of LOGITS after concatenation and averaging: {LOGITS.shape}")  # Debug: tamaño de LOGITS después de concatenar y promediar

print(LOGITS)
PREDS = LOGITS.sum(1).numpy()
print(PREDS)
real_col = df['isup_grade']
df['isup_grade'] = PREDS
print(df)

print("###############################################################")
df['isup_grade'] = df['isup_grade'].apply(lambda x: int(np.round(x)))
df[['image_id', 'isup_grade']].to_csv('submission.csv', index=False)
print(df)
print()

accuracy = calculate_accuracy_from_df(df['isup_grade'], real_col)
print(f"Accuracy: {accuracy}%")

file_name = f"accuracy_{args.model_idx}.txt"
with open(file_name, 'w') as file:
    file.write(f"Accuracy: {accuracy}%\n")

print(f"Archivo guardado como {file_name}")



