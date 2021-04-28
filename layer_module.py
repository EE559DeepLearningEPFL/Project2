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
        self.bn_param = {}
        self.conv_param = None
        
        self.dtype = dtype
        self.device = device
        
        ------------------ Linear---------------------
        self.layer_type = 'Linear'   
        self.if_batchnorm = True / False
        self.activation_type = None / 'sigmoid' / 'tanh' / 'relu'
        self.specification = (dim_in, dim_out)
        self.params = {'weight': ..., 'bias': ...}
        self.bn_param = {'eps': 1e-5, 'momentum': 0.9, ... }
        
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
        self.bn_param = {'eps': 1e-5, 'momentum': 0.9, ... }
        
        self.dtype = dtype
        self.device = device
        
        self.layer_sequence = []
    '''
    
    # constructor
    def __init__(self, 
                 layer_type, 
                 if_batchnorm=None, 
                 activation_type=None, 
                 params_shape=None, 
                 bn_param={'eps':1e-5, 'momentum':0.9}, 
                 conv_param={'padding':0, 'stride':1},
                 reg = 0.0,
                 learning_rate = 1e-3,
                 dtype=torch.float32, 
                 device=None):
        
        self.layer_type = layer_type
        self.if_batchnorm = if_batchnorm
        self.activation_type = activation_type
        self.params_shape = params_shape
        self.bn_param = bn_param
        self.conv_param = conv_param
        self.reg = reg
        self.lr = learning_rate
        self.dtype = dtype
        self.device = device
        
        self.layer_sequence = []
        self.params = {}
        
        supported_layer_type = ['Sequential', 'Linear', 'Conv2d']
        supported_activation_type = ['sigmoid', 'relu', 'tanh']
        
        assert layer_type in supported_layer_type, \
            "__init__: layer_type not supported, choose in 'Sequential', 'Linear' and 'Conv2d'."
        
        if layer_type == 'Sequential':
            assert (activation_type is None) and (params_shape is None) and (if_batchnorm is None), \
                "__init__: when layer type is Sequential, activation_type, params_shape and \
                if_batchnorm should all be None."
        else:
            assert (type(if_batchnorm) is bool), \
                "__init__: for Linear or Conv2d layer, if_batchnorm should be a boolean."
            assert (activation_type == None) or (activation_type in supported_activation_type), \
                "__init__: for activation functions, this module only support type of None, sigmoid,\
                tanh and relu."
            assert (params_shape != None) and \
                    ((type(params_shape) is tuple) or (type(params_shape) is list)), \
                "__init__: for Linear / Conv2d layers, params_shape cannot be None and should \
                be tuple or list."
            if type(params_shape) is list: self.params_shape = tuple(params_shape)
                
            if layer_type == 'Linear':
                self._initialize_Linear()
            else:
                self._initialize_Conv2d()
                
            if if_batchnorm:
                self.bn_param['if_initailized'] = False
                self.bn_param['gamma'] = None
                self.bn_param['beta'] = None
                self.bn_param['running_mean'] = None
                self.bn_param['running_var'] = None
                
    ############################################################################    
    # public methods
    ############################################################################
    
    def print_module(self):
        
        return
    
    def append(self, layer):
        assert self.layer_type == 'Sequential', \
            "sequential_append: this method is only for Sequential type module."
        self._sequential_append(layer)
        return
    
    def forward(self, input, mode='train'):
        if self.layer_type == 'Sequential':
            output, cache = self._sequential_forward(input, mode)
        else:
            output, cache = self._layer_forward(input, mode)
        return output, cache
    
    def backward(self, d_output, cache):
        if self.layer_type == 'Sequential':
            d_input, d_params = self._sequential_backward(d_output, cache)
        else:
            d_input, d_params = self._layer_backward(d_output, cache)
        return d_output, d_params
    
    def update_params(self, d_params, learning_rate):
        if self.layer_type == 'Sequential':
            self._sequential_update_params(d_params, learning_rate)
        else:
            self._layer_update_params(d_params, learning_rate)
        return
            
    ############################################################################    
    # private methods
    ############################################################################  
    '''
    module-level forward/backward/updata_params
    '''
    
    '''
    For Sequential type:
    
        _sequential_append(Module)
        output, cache = _sequential_forward(input)
        d_params = _sequential_backward(d_output, cache)
        _sequential_updata_params(d_params, learning_rate)
        
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
    
    def _sequential_append(self, layer): 
        assert self.layer_type == 'Sequential', \
            "sequential_append: this method is only for Sequential type module."
        if (not isempty(self.layer_sequence)) and (layer.layer_type == 'Conv2d'):
            assert self.layer_sequence[len(self.layer_sequence)-1].layer_type is not 'Linear', "sequential_append: Conv2d should not follow Linear layer."
        self.layer_sequence.append(layer)
        return
        
    def _sequential_forward(self, input, mode='train'):
        assert self.layer_type == 'Sequential', \
            "sequential_append: this method is only for Sequential type module."
        #
        output, cache = None, None
        return output, cache
    
    def _sequential_backward(self, d_output, cache):
        assert self.layer_type == 'Sequential', \
            "sequential_append: this method is only for Sequential type module."
        #
        d_params = None
        return d_params
    
    def _sequential_update_params(self, d_params, learning_rate):
        assert self.layer_type == 'Sequential', \
            "sequential_append: this method is only for Sequential type module."
        # for loop of layers
        return
    
    '''
    For Linear/Conv2d type:

        output, cache = _layer_forward(input)
        d_output, ... = _layer_backward(d_output, cache)
        _layer_updata_params(...)
    '''
        
    def _layer_forward(self, input, mode='train'):
        assert self.layer_type != 'Sequential', \
            "layer_forward(Linear): this method is only for non-Sequential type module."
        #
        output, cache = None, None
        return output, cache
            
    def _layer_backward(self, d_output, cache):
        assert self.layer_type != 'Sequential', \
            "layer_forward: this method is only for non-Sequential type module."
        #
        d_input = None
        d_params = {}
        return d_input, d_params
    
    def _layer_update_params(self, d_params, learning_rate):
        assert self.layer_type != 'Sequential', \
            "layer_forward: this method is only for non-Sequential type module."
        #
        return
        
 
    
    '''
    Layer forward/backward pass related:
    
        output, cache = _forward_Linear(input)
        d_input, d_weight, d_bias = _backward_Linear(d_output, cache)
        
        output, cache = _forward_Conv2d(input, conv_spec)
        d_input, d_weight, d_bias = _backward_Linear(d_output, cache)
    '''
    
    def _forward_Linear(self, input):
        input_row = torch.reshape(input, (input.size(0), -1))
        assert self.params_shape[0] == input_row.size()[1], \
            "_forward_Linear: input and parameter dimension not matched."
        output = torch.matmul(input, self.params['weight']) + self.params['bias']
        cache = input
        return output, cache
    
    def _backward_Linear(self, d_output, cache):
        input = cache
        d_params = {}
        d_input = torch.reshape(torch.matmul(d_output, self.params['weight'].t), input.size())
        d_params['weight'] = torch.matmul(torch.reshape(input, (input.size(0), -1)).t, d_output)
        d_params['bias'] = torch.sum(d_output, axis=0, keepdim=True)
        return d_input, d_params
    
    def _forward_Conv2d(self, input):
        output = None
        cache = None
        return output, cache
    
    def _backward_Conv2d(self, d_output, cache):
        d_input = None
        d_params = {}
        return d_input, d_params
    
    def _forward_batchnorm(self, input, mode):
        assert (mode == 'train') or (mode == 'test'), \
            "_forward_batchnorm: mode should be 'train' or 'test'."
        
        if not self.bn_param['if_initialized']:
            self._initialize_bn_param(input)
        
        momentum = self.bn_param['momentum']
        gamma, beta = self.bn_param['gamma'], self.bn_param['beta']
        running_mean, running_var = self.bn_param['running_mean'], self.bn_param['running_var']
                
        if mode == 'train':
            sample_mean = torch.mean(input, axis=0, keepdim=True)
            sample_var = torch.var(input, axis=0, keepdim=True)
            input_normed = (input - sample_mean)/torch.sqrt(sample_var+self.bn_params['eps'])
            running_mean = momentum*running_mean + (1-momentum)*sample_mean
            running_var = momentum*running_var + (1-momentum)*sample_var
            output = gamma*input_normed + beta
            cache = (input, input_normed, sample_mean, sample_var)
        else:
            input_normed = (input-running_mean) / torch.sqrt(running_var+self.bn_params['eps'])
            output = gamma*input_normed + beta
            cache = None
        
        self.bn_param['running_mean'], self.bn_param['running_var'] = running_mean, running_var
        
        return output, cache
    
    def _backward_batchnorm(self, d_output, cache):
        input, input_normed, sample_mean, sample_var = cache
        gamma, beta = self.bn_param['gamma'], self.bn_param['beta']
        N = input.size(0)
        
        d_input_normed = gamma*d_output
        inv_std = 1.0 / torch.sqrt(sample_var+self.bn_param['eps'])
        d_sample_var = -0.5*torch.sum(d_input_normed*(input-sample_mean), axis=0, keepdim=True)*(inv_std**3)
        d_sample_mean = -1.0*torch.sum(d_input_normed*inv_std, axis=0, keepdim=True) \
                        -2.0*d_sample_var*torch.mean(input-sample_mean, axis=0, keepdim=True)
        d_input = d_input_normed*inv_std + 2.0*d_sample_var*(input-sample_mean)/N + 1.0*d_sample_mean/N
        
        d_params = {}
        d_params['d_gamma'] = torch.sum(d_output*input_normed, axis=0, keepdim=True)
        d_params['d_beta'] = torch.sum(d_output, axis=0, keepdim=True)
        
        return d_input, d_params
        
    
    '''
    Activation forward/backward related:
        
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
        assert d_output.size() == cache.size(), \
            "_backward_sigmoid: two inputs should be of same size."
        output = cache
        d_input = d_output*output*(1-output)
        return d_input
    
    def _forward_tanh(self, input):
        output = torch.tanh(input)
        cache = output
        return output, cache
    
    def _backward_tanh(self, d_output, cache):
        assert d_output.size() == cache.size(), \
            "_backward_tanh: two inputs should be of same size."
        output = cache
        d_input = d_output*(1-torch.square(output))
        return d_input
    
    def _forward_relu(self, input):
        output = torch.maximum(input, 0)
        cache = input
        return output, cache
    
    def _backward_relu(self, d_output, cache):
        assert d_output.size() == cache.size(), \
            "_backward_relu: two inputs should be of same size."
        input = cache
        d_input = d_output*(input>0)
        return d_input
    
    '''
    Parameter initialization related:
    
        _initialize_Linear()
        _initialize_Conv2d()
        gain = _activation_gain()
    '''
 
    def _initialize_Linear(self):
        print(self.params_shape)
        assert len(self.params_shape) == 2, \
            "_initialize_Linear: for layer_type of Linear, the params_shape should be \
            a tuple of length 2 (dim_in, dim_out)."
        gain = self._activation_gain()
        std = gain * math.sqrt(2.0/(self.params_shape[0] + self.params_shape[1]))
        self.params['weight'] = torch.empty(self.params_shape[0], self.params_shape[1], dtype=self.dtype, device=self.device).normal_(0, std)
        self.params['bias'] = torch.empty(1, self.params_shape[1], dtype=self.dtype, device=self.device).normal_(0, std)
        
    
    def _initialize_Conv2d(self):
        assert (len(self.params_shape) == 3) or (len(self.params_shape) == 4), \
            '_initialize_Conv2d: for layer_type of Linear, the params_shape should \
            be a tuple of length 3 (channel_out, channel_in, kernel_size) or length \
            4 (channel_out, channel_in, kernel_height, kernel_width)'
        if len(self.params_shape) == 3:
            c_out, c_in, k_s = self.params_shape
            k_h, k_w = k_s, k_s
        else:
            c_out, c_in, k_h, k_w = self.params_shape
            
        gain = self._activation_gain()
        std = gain * math.sqrt(1.0/(c_in*k_h*k_w))
        self.params['weight'] = torch.empty(c_out, c_in, k_h, k_w, dtype=self.dtype, device=self.device).normal_(0, std)
        self.params['bias'] = torch.empty(1, c_out, dtype=self.dtype, device=self.device).normal_(0, std)
        

    def _activation_gain(self):
        '''
        gain on weight standard deviation brought by activation function
        '''
        if self.activation_type == 'sigmoid' or self.activation_type == None:
            return 1.0
        elif self.activation_type == 'tanh':
            return 5.0/3
        elif self.activation_type == 'relu':
            return math.sqrt(2.0)
        return 1.0
    
    def _initialize_bn_param(self, input):
        self.bn_param['gamma'] = torch.ones([1]+list(input.size())[1:], dtype=self.dtype, device=self.device)
        self.bn_param['beta'] = torch.zeros([1]+list(input.size())[1:], dtype=self.dtype, device=self.device)
        self.bn_param['running_mean'] = torch.zeros([1]+list(input.size())[1:], dtype=self.dtype, device=self.device)
        self.bn_param['running_var'] = torch.zeros([1]+list(input.size())[1:], dtype=self.dtype, device=self.device)
        self.bn_param['if_initialized'] = True
        return
    
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
        assert input.dim == 4, \
            '_zero_padding: This function only supports input tensor of 4-dimensional.'
        assert (type(padding) is int) or (type(padding) is tuple), \
            '_zero_padding: Wrong padding params_shape, input type should be int or tuple.'
        
        if type(padding) is int:
            output = torch.zeros(input.size(0), input.size(1), input.size(2)+2*padding, input.size(3)+2*padding, dtype=self.dtype, device=self.device)
            output[:, :, padding:padding+input.size(2), padding:padding+input.size(3)] = input
            
        else:
            assert len(padding) == 2, \
                '_zero_padding: Only accept tuple of length 2.'
            output = torch.zeros(input.size(0), input.size(1), input.size(2)+2*padding[0], input.size(3)+2*padding[1], dtype=self.dtype, device=self.device)
            output[:, :, padding[0]:padding[0]+input.size(2), padding[1]:padding[1]+input.size(3)] = input
        
        return output.contiguous()
    