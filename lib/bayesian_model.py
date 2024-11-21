import torch
import torch.nn as nn
import torchbnn as bnn
import numpy as np
import torch.optim as optim


class BayesianENet(nn.Module):
    def __init__(self, modelo, in_features, output_dim=5, hidden_dim=128):
        super(BayesianENet, self).__init__()
        self.model_num = modelo
        if self.model_num == 0:
            self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features, out_features=16)
            self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=8)
            self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=output_dim)

        elif self.model_num == 1:
            self.bayesian_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features,
                                               out_features=output_dim)

        elif self.model_num == 2:
            self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features, out_features=32)
            self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=32, out_features=16)
            self.dropout = nn.Dropout(0.5)
            self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=8)
            self.fc4 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=output_dim)
            
    def forward(self, x):
        if self.model_num == 0:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.model_num == 1:
            x = self.bayesian_fc(x)
        elif self.model_num == 2:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
        return x

    def setup_optimizer_and_criterion(self, learning_rate):
        # Configuración del optimizador
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        # loss_fn = torch.nn.CrossEntropyLoss()  # Función de pérdida
        loss_fn = nn.MSELoss()

        return optimizer, criterion, loss_fn

    # Método para cargar pesos preentrenados
    def load_pretrained_weights(self, path_pretrained_weights='/data/hook/myfc_weights.npy',
                                path_pretrained_bias='/data/hook/myfc_bias.npy'):
        pretrained_weights = np.load(path_pretrained_weights)  # Cargamos los pesos preentrenados
        pretrained_bias = np.load(path_pretrained_bias)
        # Cargamos los pesos y bias en los parámetros de la media (mu) de la capa bayesiana
        with torch.no_grad():
            self.bayesian_fc.weight_mu.copy_(torch.tensor(pretrained_weights))
            self.bayesian_fc.bias_mu.copy_(torch.tensor(pretrained_bias))
            # TODO: modificacion sigma a 0 PRUEBA


    def summary(self, input_size):
        """
        Imprime un resumen del modelo incluyendo las capas, dimensiones de salida y parámetros.

        Parameters
        ----------
        input_size : tuple
            Dimensiones de entrada al modelo (sin incluir el tamaño del batch).
        """
        print(f"{'Layer':<20}{'Output Shape':<25}{'Param #':<15}")
        print("=" * 60)

        total_params = 0
        x = torch.zeros(1, *input_size)  # Simula una entrada
        for name, module in self.named_children():
            x = module(x)
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            print(f"{name:<20}{str(list(x.shape)):<25}{num_params:<15}")

        print("=" * 60)
        print(f"Total Params: {total_params}")