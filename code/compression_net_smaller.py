import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.modules.utils import _pair

from torch.autograd import Function
from IPython import embed

class ConvRNN(nn.Module):
    def __init__(self):
        super(ConvRNN, self).__init__()

class ConvLSTM(ConvRNN):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

class ConvGRU(ConvRNN):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvGRU,self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        
        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)
        gate_channels = 2 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)
        
        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.conv_ih_new = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh_new = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()
        self.conv_ih_new.reset_parameters()
        self.conv_hh_new.reset_parameters()

    def forward(self, input, hidden):
        #input 16*16, h1 8*8
        h1, _ = hidden

        try:
            gates = self.conv_ih(input) + self.conv_hh(h1)
        except:
            import IPython
            IPython.embed()
        rt, zt = gates.chunk(2, 1)

        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(zt)
        gated_hidden = torch.mul(reset_gate, h1)

        gate_new = self.conv_ih_new(input) + self.conv_hh_new(gated_hidden)
        h_tilde= F.tanh(gate_new)

        next_h = torch.mul(update_gate, h_tilde) + torch.mul(1-update_gate, h1)
        return next_h, None

class SignFunction(Function):

    def __init__(self):
        super(SignFunction, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        # Apply quantization noise while only training
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class Sign(nn.Module):
    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)

class CompressionEncoder(nn.Module):

    def __init__(self, rnn_type='LSTM'):
        super(CompressionEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)

        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        if rnn_type == 'LSTM':
            ConvRNN = ConvLSTM
        else:
            ConvRNN = ConvGRU
        
        self.rnn1 = ConvRNN(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvRNN(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvRNN(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn4 = ConvRNN(
            512,
            1024,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, x, h1, h2, h3, h4):
        x = self.conv1(x)
        h1_new = self.rnn1(x, h1)
        x = h1_new[0]
        h2_new = self.rnn2(x, h2)
        x = h2_new[0]
        h3_new = self.rnn3(x, h3)
        x = h3_new[0]
        h4_new = self.rnn4(x, h4)
        x = h4_new[0]

        return x, h1_new, h2_new, h3_new, h4_new

class CompressionBinarizer(nn.Module):

    def __init__(self):
        super(CompressionBinarizer, self).__init__()
        self.conv1 = nn.Conv2d(1024, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.sign = Sign()

    def forward(self,x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.sign(x)
        return x

class CompressionDecoder(nn.Module):

    def __init__(self, rnn_type='LSTM'):
        super(CompressionDecoder, self).__init__()

        self.conv1 = nn.Conv2d(32, 1024, kernel_size=1, stride=1, padding=0, bias=False)

        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        if rnn_type == 'LSTM':
            ConvRNN = ConvLSTM
        else:
            ConvRNN = ConvGRU
       
        self.rnn0 = ConvRNN(
            1024,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        
        self.rnn1 = ConvRNN(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvRNN(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvRNN(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.rnn4 = ConvRNN(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.conv2 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self,x,h0, h1,h2,h3,h4):
        x = self.conv1(x)
        
        h0_new = self.rnn0(x, h0)
        x = h0_new[0]
        x = F.pixel_shuffle(x, 2)
        
        h1_new = self.rnn1(x,h1)
        x = h1_new[0]
        x = F.pixel_shuffle(x, 2)

        h2_new = self.rnn2(x,h2)
        x = h2_new[0]
        x = F.pixel_shuffle(x, 2)

        h3_new = self.rnn3(x,h3)
        x = h3_new[0]
        x = F.pixel_shuffle(x, 2)

        h4_new = self.rnn4(x,h4)
        x = h4_new[0]
        x = F.pixel_shuffle(x, 2)

        x = self.conv2(x)
        x = F.tanh(x) / 2

        return x, h0_new, h1_new, h2_new, h3_new, h4_new
