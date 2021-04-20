import torch
import math
torch.set_grad_enabled(False)

class Module(object):
    
    def __init__(self, layer_type, activation_type, specification, dtype=torch.float32, device=None):
        
        self.layer_type = layer_type
        self.activation_type = activation_type
        self.specification = specification
        self.dtype = dtype
        self.device = device
        self.layer_sequence = []
        self.params = {}
        
        if layer_type == 'Sequential':
            assert (activation_type is None) && (specification is None), '__init__: when layer type is Sequential, activation_type and specification should both be None.'
        else:
            assert (layer_type == 'Linear') || (layer_type == 'Conv2d'), '__init__: for single layers, this module only support type of Linear and Conv2d.'
            assert (activation_type == None) || (activation_type == 'sigmoid') || (activation_type == 'tanh') || (activation_type == 'relu'), '__init__: for activation layers, this module only support type of None, sigmoid, tanh and relu.'         
            if layer_type == 'Linear':
                self._initialize_Linear()
            else:
                self._initialize_Conv2d()
        
        
    def sequential_append(self, layer):
        assert self.layer_type == 'Sequential', 'sequential_appen: this methon is only for Sequential type module.'
        self.layer_sequence.append(layer)
        
        
    def _initialize_Linear(self):
        assert (type(self.specification) is tuple) && (len(self.specification) == 2), '_initialize_Linear: for layer_type of Linear, the specification should be a tuple of length 2 (dim_in, dim_out).'
        gain = self._activation_gain()
        std = gain * math.sqrt(2.0/(specification[0], specification[1]))
        params['weight'] = torch.empty(specification[0], specification[1], dtype=self.dtype, device=self.device).normal_(0, std)
        params['bias'] = torch.empty(1, specification[1], dtype=self.dtype, device=self.device).normal_(0, std)
        
    def _initialize_Conv2d(self):
        assert (type(self.specification) is tuple), '_initialize_Conv2d: for layer_type of Conv2d, the specification should be a tuple.'
        assert (len(self.specification) == 3) || (len(self.specification) == 4), '_initialize_Conv2d: for layer_type of Linear, the specification should be a tuple of length 3 (channel_out, channel_in, kernel_height, kernel_width).'
        if len(self.specification) == 3:
            c_out, c_in, k_s = self.specification
            k_h, k_w = k_s, k_s
        else:
            c_out, c_in, k_h, k_w = self.specification
            
        gain.self._activation_gain()
        std = gain * math.sqrt(1.0/(c_in*k_h*k_w))
        params['weight'] = torch.empty(c_out, c_in, k_h, k_w, dtype=self.dtype, device=self.device).normal_(0, std)
        params['bias'] = torch.empty(1, c_out, dtype=self.dtype, device=self.device).normal_(0, std)
        

    def _activation_gain(self):
        '''
        gain on weight standard deviation brought by activation function
        '''
        if self.activation == 'sigmoid' or self.activation == None:
            return 1.0
        elif self.activation == 'tanh':
            return 5.0/3
        elif self.activation == 'relu':
            return math.sqrt(2.0)
        return 1.0
            
    
    def _zero_padding(self, input, padding):
        '''
        Zero padding:
        
        input: 4-dimensional, (num_samples, channel, width, height)
        padding: int or tuple(length=2), e.g. 2, (2, 3)
        
        output: 4-dimensional, (num_samples, channel, width+..., height+...)
        '''
        assert input.dim == 4, '_zero_padding: This function only supports input tensor of 4-dimensional.'
        assert (type(padding) is int) || (type(padding) is tuple), '_zero_padding: Wrong padding specification, input type should be int or tuple.'
        
        if type(padding) is int:
            output = torch.zeros(input.size(0), input.size(1), input.size(2)+2*padding, input.size(3)+2*padding, dtype=self.dtype, device=self.device)
            output[:, :, padding:padding+input.size(2), padding:padding+input.size(3)] = input
            
        else:
            assert len(padding) == 2, '_zero_padding: Only accept tuple of length 2.'
            output = torch.zeros(input.size(0), input.size(1), input.size(2)+2*padding[0], input.size(3)+2*padding[1], dtype=self.dtype, device=self.device)
            output[:, :, padding[0]:padding[0]+input.size(2), padding[1]:padding[1]+input.size(3)] = input
        
        return output.contiguous()
    