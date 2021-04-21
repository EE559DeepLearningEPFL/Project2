import torch
import math
torch.set_grad_enabled(False)

class Module(object):
    
    '''
    Example:
        ---------------- Sequential --------------------
        self.layer_type = 'Sequential'
        self.layer_sequence = [layer1, layer2, ...]
        
        self.if_batchnorm = None
        self.activation_type = None
        self.specification = None
        self.params = {}
        self.conv_param = None
        
        self.dtype = dtype
        self.device = device
        
        ------------------ Linear---------------------
        self.layer_type = 'Linear'   
        self.if_batchnorm = True / False
        self.activation_type = None / 'sigmoid' / 'tanh' / 'relu'
        self.specification = (dim_in, dim_out)
        self.params = {'weight': ..., 'bias': ...}
        
        self.dtype = dtype
        self.device = device
        
        self.conv_param = None 
        self.layer_sequence = []
        
        ---------------- Conv2d --------------------
        self.layer_type = 'Conv2d'
        self.if_batchnorm = True / False
        self.activation_type = None / 'sigmoid' / 'tanh' / 'relu'
        self.specification = (channel_out, channel_in, kernel_size) / (channel_out, channel_in, kernel_height, kernel_width)
        self.conv_param = {'padding': padding, 'stride': stride}
        self.params = {'weight': ..., 'bias': ...}
        
        self.dtype = dtype
        self.device = device
        
        self.layer_sequence = []
    '''
    
    # constructor
    def __init__(self, layer_type, if_batchnorm, activation_type, 
                 specification, dtype=torch.float32, device=None):
        
        self.layer_type = layer_type
        self.if_batchnorm = if_batchnorm
        self.activation_type = activation_type
        self.specification = None
        self.dtype = dtype
        self.device = device
        self.layer_sequence = []
        self.params = {}
        self.conv_param = None
        
        supported_layer_type = ['Sequential', 'Linear', 'Conv2d']
        supported_activation_type = ['sigmoid', 'relu', 'tanh']
        
        assert layer_type in supported_layer_type, "__init__: layer_type not supported, choose in 'Sequential', 'Linear' and 'Conv2d'."
        
        if layer_type == 'Sequential':
            assert (activation_type is None) && (specification is None) && (if_batchnorm is None), "__init__: when layer type is Sequential, activation_type and specification should both be None."
        
        else:
            assert (type(if_batchnorm) is bool), "__init__: for Linear or Conv2d layer, if_batchnorm should be a boolean."
            assert (activation_type == None) || (activation_type in supported_activation_type), "__init__: for activation layers, this module only support type of None, sigmoid, tanh and relu."
            if layer_type == 'Linear':
                self.specification = specification
                self._initialize_Linear()
            else:
                assert (type(specification) is tuple) && (len(specification)==2), "__init__: when layer_type is Conv2d, the input specification should be ((channel_out, channel_in, ...), {'padding': ..., 'stride': ...})"
                self.specification = specification[0]
                self.conv_param = specification[1]
                self._initialize_Conv2d()
                
    ############################################################################    
    # public methods
    ############################################################################
    
    '''
    For Sequential type:
    
        sequential_append(Module)
        output, cache = sequential_forward(input)
        d_params = sequential_backward(d_output, cache)
        sequential_updata_params(d_params, learning_rate)
        
    NOTE: 
    
        (Updated: 2021/04/21)  
        
        Propose format of cache:
            type: list of tuple/tensor
            order of dicts: [cache_layer1, cache_layer2, ..., cache_layer5]
        
        Propose output of sequential_backward(d_output, cache):
            type: list of dictionary
            order of dicts: reverse, e.g. [dict_layer5, dict_layer4, ..., dict_layer1]
            each dict: {'d_weight': ..., 'd_bias': ...}
    '''
    
    def sequential_append(self, layer): 
        assert self.layer_type == 'Sequential', "sequential_append: this method is only for Sequential type module."
        self.layer_sequence.append(layer)
        return
        
    def sequential_forward(self, input):
        assert self.layer_type == 'Sequential', "sequential_append: this method is only for Sequential type module."
        #
        output, cache = None, None
        return output, cache
    
    def sequential_backward(self, d_output, cache):
        assert self.layer_type == 'Sequential', "sequential_append: this method is only for Sequential type module."
        #
        d_params = None
        return d_params
    
    def sequential_updata_params(self, d_params, learning_rate):
        assert self.layer_type == 'Sequential', "sequential_append: this method is only for Sequential type module."
        # for loop of layers
        return
    
    '''
    For Linear/Conv2d type:

        output, cache = layer_forward(input)
        d_output, ... = layer_backward(d_output, cache)
        layer_updata_params(...)
    '''
        
    def layer_forward(self, input):
        assert self.layer_type != 'Sequential', "layer_forward(Linear): this method is only for non-Sequential type module."
        #
        output, cache = None, None
        return output, cache
            
    def layer_backward(self, d_output, cache):
        assert self.layer_type != 'Sequential', "layer_forward: this method is only for non-Sequential type module."
        #
        d_input = None
        d_params = {}
        return d_input, d_params
    
    def layer_update_params(self, d_params, learning_rate):
        assert self.layer_type != 'Sequential', "layer_forward: this method is only for non-Sequential type module."
        #
        return
        
        
    ############################################################################    
    # private methods
    ############################################################################   
    
    '''
    Forward/Backward pass related:
    
        output, cache = _forward_Linear(input)
        d_input, d_weight, d_bias = _backward_Linear(d_output, cache)
        
        output, cache = _forward_Conv2d(input, conv_spec)
        d_input, d_weight, d_bias = _backward_Linear(d_output, cache)
        
        output, cache = _forward_relu(input)
        d_input = _backward_relu(d_output, cache)
            
        output, cache = _forward_tanh(input)
        d_input = _backward_tanh(d_output, cache)
           
        output, cache = _forward_sigmoid(input)
        d_input = _backward_sigmoid(d_output, cache)
        ...
    '''
    
    def _forward_sigmoid(self, input):
        output = torch.sigmoid(input)
        cache = output
        return output, cache
    
    def _backward_sigmoid(self, d_output, cache):
        assert d_output.size() == cache.size(), "_backward_sigmoid: two inputs should be of same size."
        output = cache
        d_input = d_output*output*(1-output)
        return d_input
    
    def _forward_tanh(self, input):
        output = torch.tanh(input)
        cache = output
        return output, cache
    
    def _backward_tanh(self, d_output, cache):
        assert d_output.size() == cache.size(), "_backward_sigmoid: two inputs should be of same size."
        output = cache
        d_input = d_output*(1-torch.square(output))
        return d_input
    
    '''
    Parameter initialization related:
    
        _initialize_Linear()
        _initialize_Conv2d()
        gain = _activation_gain()
    '''
 
    def _initialize_Linear(self):
        assert (type(self.specification) is tuple) && (len(self.specification) == 2), '_initialize_Linear: for layer_type of Linear, the specification should be a tuple of length 2 (dim_in, dim_out).'
        gain = self._activation_gain()
        std = gain * math.sqrt(2.0/(specification[0], specification[1]))
        params['weight'] = torch.empty(specification[0], specification[1], dtype=self.dtype, device=self.device).normal_(0, std)
        params['bias'] = torch.empty(1, specification[1], dtype=self.dtype, device=self.device).normal_(0, std)
        
    
    def _initialize_Conv2d(self):
        assert (type(self.specification) is tuple), '_initialize_Conv2d: for layer_type of Conv2d, the specification should be a tuple.'
        assert (len(self.specification) == 3) || (len(self.specification) == 4), '_initialize_Conv2d: for layer_type of Linear, the specification should be a tuple of length 3 (channel_out, channel_in, kernel_size) or length 4 (channel_out, channel_in, kernel_height, kernel_width)'
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
    
    
    '''
    Other functional methods:
    
    
    '''
            
    
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
    