from time import time
# from matplotlib import backend_bases
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from config import *

gpu = 0
device = torch.device(f'cuda:{str(gpu)}' if torch.cuda.is_available() else 'cpu')

class BRITS(nn.Module):
    def __init__(self, feature_dim, rnn_dim):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super(BRITS, self).__init__()

        self.rnn_dim = rnn_dim
        self.feature_dim = feature_dim

        self.forward_rnn = nn.LSTMCell(feature_dim * 2, rnn_dim).to(device)  # batch_first guarantees the order of output = (B,S,F)
        self.backward_rnn = nn.LSTMCell(feature_dim * 2, rnn_dim).to(device)  # batch_first guarantees the order of output = (B,S,F)

        self.forward_linear = nn.Linear(rnn_dim, feature_dim).to(device)
        self.backward_linear = nn.Linear(rnn_dim, feature_dim).to(device)

        self.forward_linear_beta = torch.nn.Linear(feature_dim * 2, feature_dim * 3).to(device)
        self.backward_linear_beta = torch.nn.Linear(feature_dim * 2, feature_dim * 3).to(device) 

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.forward_linear_z = FeatureRegression(feature_dim).to(device)
        self.backward_linear_z = FeatureRegression(feature_dim).to(device)

        self.m = nn.ReLU().to(device)

        self.linear_layer_rnn = nn.Linear(feature_dim, rnn_dim).to(device)
        self.linear_layer_rnn_back = nn.Linear(feature_dim, rnn_dim).to(device)

        self.linear_layer_beta = nn.Linear(rnn_dim, feature_dim).to(device)
        self.linear_layer_beta_back = nn.Linear(rnn_dim, feature_dim).to(device)

        self.linear_layer_test = nn.Linear(feature_dim*3, feature_dim).to(device)

    def forward(self, x, masking, time_lag):
        # x = B x S x 44
        # masking = B x S x 44
        # time_lag = B x S x 44

        x = x.permute(1, 0, 2).to(device)
        masking = masking.permute(1, 0, 2).to(device)
        time_lag = time_lag.permute(1, 0, 2).to(device)

        bs = x.shape[1]
        seq_len = x.shape[0]
 
        h = Variable(torch.zeros((bs, self.rnn_dim)))
        c = Variable(torch.zeros((bs, self.rnn_dim)))
        r_rnn = torch.exp(-self.m(self.linear_layer_rnn(time_lag)))  # decay_rate = B x S x rnn_dim
        r_beta = self.linear_layer_beta(r_rnn)

        if torch.cuda.is_available():
            h, c, r_rnn, r_beta, masking = h.to(device), c.to(device), r_rnn.to(device), r_beta.to(device), masking.to(device)

        ## Forward imputation

        output_list = []
        z_list = []
        c_list = []
        for j in range(seq_len):
            output = self.forward_linear(h) # B x 44

            x_c = (masking[j] * x[j] + (1 - masking[j]) * output.squeeze(0))

            z_value = self.forward_linear_z(x_c)

            beta = self.sigmoid(self.forward_linear_beta(torch.cat([masking[j], r_beta[j]], dim=-1)))
            beta = self.linear_layer_test(beta)

            c_value = beta * z_value + (1 - beta) * output

            c_c = (masking[j] * x[j] + (1 - masking[j]) * c_value.squeeze(0))

            input_data = torch.cat([c_c, masking[j]], dim=1)

            h, c = self.forward_rnn(input_data, (h * r_rnn[j], c))

            output_list.append(output.squeeze(0))
            z_list.append(z_value)
            c_list.append(c_value)

        output_list = torch.stack(output_list).float().squeeze()
        z_list = torch.stack(z_list).float().squeeze()
        c_list = torch.stack(c_list).float().squeeze()

        ## Backward imputation

        backward_output_list = []
        backward_z_list = []
        backward_c_list = []

        h = Variable(torch.zeros((bs, self.rnn_dim)))
        c = Variable(torch.zeros((bs, self.rnn_dim)))
        r_rnn = torch.exp(-self.m(self.linear_layer_rnn_back(time_lag)))  # decay_rate
        r_beta = self.linear_layer_beta_back(r_rnn)

        if torch.cuda.is_available():
            h, c, r_rnn, r_beta, masking = h.to(device), c.to(device), r_rnn.to(device), r_beta.to(device), masking.to(device)

        for j in reversed(range(seq_len)):  # x : seq_len x batch_size x feature_dim
            output = self.backward_linear(h)

            x_c = (masking[j] * x[j] + (1 - masking[j]) * output.squeeze(0))

            z_value = self.backward_linear_z(x_c)

            beta = self.sigmoid(self.backward_linear_beta(torch.cat([masking[j], r_beta[j]], dim=-1)))
            beta = self.linear_layer_test(beta)

            c_value = beta * z_value + (1 - beta) * output

            c_c = (masking[j] * x[j] + (1 - masking[j]) * c_value.squeeze(0))

            input_data = torch.cat([c_c, masking[j]], dim=1) 

            h, c = self.backward_rnn(input_data, (h * r_rnn[j], c))

            backward_output_list.append(output.squeeze(0))
            backward_z_list.append(z_value)
            backward_c_list.append(c_value)

        backward_output_list = torch.stack(list(reversed(backward_output_list))).float().squeeze()
        backward_z_list = torch.stack(list(reversed(backward_z_list))).float().squeeze()
        backward_c_list = torch.stack(list(reversed(backward_c_list))).float().squeeze()

        output_list = output_list.permute(1,0,2)
        z_list = z_list.permute(1,0,2)
        c_list = c_list.permute(1,0,2)

        backward_output_list = backward_output_list.permute(1,0,2)  # S x B x 44
        backward_z_list = backward_z_list.permute(1,0,2)  # S x B x 44
        backward_c_list = backward_c_list.permute(1,0,2)  # S x B x 44

        return output_list, z_list, c_list, backward_output_list, backward_z_list, backward_c_list

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)

        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h