import sys
import os.path as osp

import torch
from torch.autograd import Variable

from tcn import TemporalConvNet


class LSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, batch_size, device,
                 lstm_size=100, lstm_layers=3, dropout=0):
        """
        Simple LSTM network
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(LSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = out_size
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = torch.nn.LSTM(self.input_size, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size*5, self.output_size)
        self.hidden = self.init_weights()

    def forward(self, input, hidden=None):
        output, self.hidden = self.lstm(input, self.init_weights())
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)


class BilinearLSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, batch_size, device,
                 lstm_size=100, lstm_layers=3, dropout=0):
        """
        LSTM network with Bilinear layer
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(BilinearLSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.output_size = out_size
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.bilinear = torch.nn.Bilinear(self.input_size, self.input_size, self.input_size * 4)
        self.lstm = torch.nn.LSTM(self.input_size * 5, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size + self.input_size * 5, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size * 5, self.output_size)
        self.hidden = self.init_weights()

    def forward(self, input):
        input_mix = self.bilinear(input, input)
        input_mix = torch.cat([input, input_mix], dim=2)
        output, self.hidden = self.lstm(input_mix, self.init_weights())
        output = torch.cat([input_mix, output], dim=2)
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)


class TCNSeqNetwork(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, layer_channels, dropout=0.2):
        """
        Temporal Convolution Network with PReLU activations
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_channel: num. channels in input
        :param output_channel: num. channels in output
        :param kernel_size: size of convolution kernel (must be odd)
        :param layer_channels: array specifying num. of channels in each layer
        :param dropout: dropout probability
        """

        super(TCNSeqNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.num_layers = len(layer_channels)

        self.tcn = TemporalConvNet(input_channel, layer_channels, kernel_size, dropout)
        self.output_layer = torch.nn.Conv1d(layer_channels[-1], output_channel, 1)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.net = torch.nn.Sequential(self.tcn, self.output_dropout, self.output_layer)
        self.init_weights()

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.net(out)
        return out.transpose(1, 2)

    def init_weights(self):
        self.output_layer.weight.data.normal_(0, 0.01)
        self.output_layer.bias.data.normal_(0, 0.001)

    def get_receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)

class SpatialEncoderLayer(torch.nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SpatialEncoderLayer, self).__init__()
        self.conv11_0 = torch.nn.Conv1d(input_channels, input_channels, 1)
        self.conv11_1 = torch.nn.Conv1d(input_channels, input_channels, 1)
        self.conv11_2 = torch.nn.Conv1d(input_channels*2, input_channels, 1)
        self.conv11_3 = torch.nn.Conv1d(input_channels, input_channels, 1)
        self.conv11_4 = torch.nn.Conv1d(input_channels, input_channels, 1)
        self.conv33 = torch.nn.Conv1d(input_channels, input_channels, 3, groups=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2) 
        self.batch_norm_0 = torch.nn.BatchNorm1d(input_channels) 
        self.batch_norm_1 = torch.nn.BatchNorm1d(input_channels) 
        self.batch_norm_2 = torch.nn.BatchNorm1d(input_channels) 
        self.global_self_attention = torch.nn.MultiheadAttention(6, 1) 
        self.attention = torch.nn.MultiheadAttention(6, 1, batch_first=True)


    def forward(self, input_vector):

        # 1x1 conv
        x = self.conv11_0(input_vector)
        x = self.batch_norm_0(x)
        x = self.relu(x)
        # 3x3 local self attention
        c1 = self.conv33(x)
        v = self.conv11_1(input_vector)
        t = torch.concat([c1 , x], 1)
        t = self.conv11_2(t)
        t = self.relu(t)
        c2 = torch.mul(t, v)
        c1_t = c1.transpose(1, 2).contiguous()
        c2_t = c2.transpose(1, 2).contiguous()
        y, _ = self.attention(c1_t, c2_t, c2_t)
        y = y.transpose(1, 2).contiguous()
        # 1x1 conv
        y = self.conv11_3(y)
        y = self.batch_norm_1(y)
        y = self.relu(y)
        # 1x1 global self attention
        y = y.transpose(1, 2).contiguous()
        y, _ = self.global_self_attention(y, y, y)
        y = y.transpose(1, 2).contiguous()
        y = self.conv11_4(y)
        y = self.batch_norm_2(y)
        # add & ReLU
        y = y + input_vector
        y = self.relu(y)

        return y

class SpatialEncoder(torch.nn.Module):
    def __init__(self, input_channels, Ns):
        super().__init__()
        self.layers = torch.nn.ModuleList([SpatialEncoderLayer(input_channels, input_channels) for _ in range(Ns)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.layers):
            x = self.layers[i](x)
        return x
class CTINNetwork(torch.nn.Module):
    def __init__(self, input_channels=6, out_channels=2, batch_size=100, window_length=400, Ns=1, Nt=4 , device=0):
        """
        Contextual Transformer 

        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]
        """
        super(CTINNetwork, self).__init__()
        self.input_channels = input_channels
        self.output_channels = out_channels
        self.batch_size = batch_size
        self.device = device
        self.Ns = Ns
        self.Nt = Nt
        self.window_length = window_length

        self.spatial_embedder = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_channels, self.input_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(self.input_channels),
            torch.nn.Linear(self.window_length, self.window_length)
            ) 
        self.temporal_embedder = BilinearLSTMSeqNetwork(self.input_channels, self.input_channels, self.batch_size, self.device) 
        self.spatial_encoder = SpatialEncoder(self.input_channels, Ns)
        # Temporal decoder
        self.temporal_decoder_layer = torch.nn.TransformerDecoderLayer(self.window_length, 8)
        self.temporal_decoder = torch.nn.TransformerDecoder(self.temporal_decoder_layer, Nt)


        # Linear Layers
        self.linear_linear_vel = torch.nn.Sequential(
            torch.nn.Linear(self.input_channels, 4),
            torch.nn.BatchNorm1d(self.window_length),
            torch.nn.ReLU(),
            torch.nn.Linear(4, self.output_channels),
            torch.nn.LayerNorm(self.output_channels)
        ) # Linear layer for linear velocity output 
        self.linear_vel_cov = torch.nn.Sequential(
            torch.nn.Linear(self.input_channels, 4),
            torch.nn.BatchNorm1d(self.window_length),
            torch.nn.ReLU(),
            torch.nn.Linear(4, self.output_channels),
            torch.nn.ReLU() # covariance cannot be negative
        ) # Linear layer for velocity covariance output
    
    def forward(self, input):
        input_t = input.transpose(1, 2).contiguous()
        spatial_embedding = self.spatial_embedder(input_t)
        temporal_embedding = self.temporal_embedder(input)
        z = self.spatial_encoder(spatial_embedding)
        temporal_embedding_t = temporal_embedding.transpose(1, 2).contiguous()
        h = self.temporal_decoder(temporal_embedding_t, z)
        h = h.transpose(1, 2).contiguous()
        linear_vel_output = self.linear_linear_vel(h)
        linear_cov_output = self.linear_vel_cov(h)
        return linear_vel_output, linear_cov_output