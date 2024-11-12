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
            self.sigmoid = nn.Sigmoid()

        elif self.model_num == 1:
            self.bayesian_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features,
                                               out_features=output_dim)

        elif self.model_num == 2:
            self.bayesian_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features,
                                               out_features=output_dim)

        else:
            self.bayesian_fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_features,
                                                out_features=hidden_dim)
            self.bayesian_fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim,
                                                out_features=hidden_dim)
            self.bayesian_fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim,
                                                out_features=output_dim)

    def forward(self, x):
        if self.model_num == 0:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
        elif self.model_num == 1:
            x = self.bayesian_fc(x)
        elif self.model_num == 2:
            x = self.bayesian_fc(x)
            return torch.softmax(x, dim=-1)
        else:
            x = torch.relu(self.bayesian_fc1(x))
            x = torch.relu(self.bayesian_fc2(x))
            x = self.bayesian_fc3(x)
            return torch.softmax(x, dim=-1)
        return x

    def setup_optimizer_and_criterion(self, learning_rate):
        # Configuración del optimizador
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if self.model_num in [1, 2, 3]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
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
