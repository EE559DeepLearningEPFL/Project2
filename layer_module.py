import torch
import math
torch.set_grad_enabled(False)

class Module(object):
    
    # constructor
    def __init__(self, 
                 layer_type, 
                 if_batchnorm=None, 
                 activation_type=None, 
                 params_shape=None, 
                 bn_param={'eps':1e-5, 'momentum':0.9}, 
                 conv_param={'padding':0, 'stride':1},
                 dtype=torch.float32, 
                 device=None):
        
        # specify the type of this module, 'Sequential'/'Linear'/'Conv2d'.
        self.layer_type = layer_type
        
        # specify if this module includes batch normaization.
        self.if_batchnorm = if_batchnorm
        
        # specify what activation function to use, 'relu'/'tanh'/'sigmoid'/None.
        self.activation_type = activation_type
        
        # specify the shape of the weight matrix/array
        # for 'Linear': (dim_in, dim_out)
        # for 'Conv2d': (channel_out, channel_in, kernel_height, kernel_width)
        self.params_shape = params_shape
        
        # specify batchnorm related parameters, 
        # include: 'momentum', 'eps', 'gamma', 'beta', 'running_mean', 'running_variance'
        self.bn_param = bn_param.copy()
        
        # specify convolution related parameters, 
        # include: 'padding', 'stride'
        self.conv_param = conv_param.copy()
        
        self.dtype = dtype
        self.device = device
        
        # initialize the list to store layer sequence, 
        # for 'Sequential' type only
        self.layer_sequence = []
        
        # initialize dictionary to store weight and bias,
        # for 'Linear' or 'Conv2d'
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
                
            # initialize weight and bias
            if layer_type == 'Linear':
                self._initialize_Linear()
            else:
                self._initialize_Conv2d()
            
            # set up batch normalization related bariables, has to initialize when there's input.
            if if_batchnorm:
                self.bn_param['if_initialized'] = False
                self.bn_param['gamma'] = None
                self.bn_param['beta'] = None
                self.bn_param['running_mean'] = None
                self.bn_param['running_var'] = None
                
    ############################################################################ 
    ############################################################################ 
    ########################## PUBLIC METHODS ##################################
    ############################################################################
    ############################################################################
    
    '''
    To print the information of the module:
        print_module()
        
    Append a module to the end of the sequence, for 'Sequential' only:
        append(Module)
        
    Forward pass and store backpropagation related data:
        output, cache = forward(input, mode='train')
    
    Compute loss and its gradient w.r.t. the output:
        loss, d_loss = loss(output, target, regularization)
        
    Backward pass:
        d_input, d_params = backward(d_output, cache)
        
    Update the parameters by gradient descent:
        update_params(d_params, learning_rate, regularization)
    '''
    
    def print_module(self): 
        if self.layer_type == 'Sequential':
            for ii, layer in enumerate(self.layer_sequence):
                print("Layer {}:".format(ii+1))
                layer._layer_print()
                print(" ")
        else:
            self._layer_print()
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
    
    def loss(self, output, target, regularization):
        loss = self._compute_loss(output, target, regularization)
        d_loss = self._compute_loss_gradient(output, target)
        return loss, d_loss
    
    def backward(self, d_output, cache):
        if self.layer_type == 'Sequential':
            d_input, d_params = self._sequential_backward(d_output, cache)
        else:
            d_input, d_params = self._layer_backward(d_output, cache)
        return d_input, d_params
    
    def update_params(self, d_params, learning_rate, regularization):
        if self.layer_type == 'Sequential':
            self._sequential_update_params(d_params, learning_rate, regularization)
        else:
            self._layer_update_params(d_params, learning_rate, regularization)
        return
    
    ############################################################################ 
    ############################################################################ 
    ########################## PRIVATE METHODS #################################
    ############################################################################
    ############################################################################
    '''
    module-level forward/backward/updata_params
    '''
    
    '''
    For Sequential type:
    
        _sequential_append(Module)
        
        output, cache = _sequential_forward(input)
        
        d_input, d_params = _sequential_backward(d_output, cache)
        
        _sequential_updata_params(d_params, learning_rate)
    '''
    
    def _sequential_append(self, layer): 
        assert layer.layer_type != 'Sequential', \
            "_sequential_append: cannot append sequential type to the sequence."
        
        if (len(self.layer_sequence) != 0) and (layer.layer_type == 'Conv2d'):
            assert self.layer_sequence[len(self.layer_sequence)-1].layer_type != 'Linear', \
                "_sequential_append: Conv2d should not follow Linear layer."
        
        self.layer_sequence.append(layer)
        return
        
    def _sequential_forward(self, input, mode='train'):
        '''
        cache: list of cache from each module, in the order of layers
                e.g. [cache_layer_1, ..., cache_layer_N]
        '''
        cache = []
        output = input
        for layer in self.layer_sequence:
            output, cache_layer = layer._layer_forward(output, mode)
            cache.append(cache_layer)
            
        return output, cache
    
    def _sequential_backward(self, d_output, cache):
        '''
        d_params: list of d_params from each module, in the order of layers
                e.g. [d_params_layer_1, ..., d_params_layer_N]
        '''
        d_params = []
        d_input = d_output
        # iterate in the inverse order
        for ii in range(len(self.layer_sequence)-1, -1, -1):
            d_input, d_params_layer = self.layer_sequence[ii]._layer_backward(d_input, cache[ii])
            # insert the d_params_layer to top of list
            d_params = [d_params_layer] + d_params
            
        return d_input, d_params
    
    def _sequential_update_params(self, d_params, learning_rate, regularization):
        for ii, layer in enumerate(self.layer_sequence):
            layer._layer_update_params(d_params[ii], learning_rate, regularization)
        return
    
    '''
    For Linear/Conv2d type:
    
        _layer_print()
        
        output, cache = _layer_forward(input)
        
        d_output, d_params = _layer_backward(d_output, cache)
        
        _layer_updata_params(d_params, learning_rate, regularization)
        
    '''
    
    def _layer_print(self):
        print("layer_type={}, if_batchnorm={}, activation_type={},".format(
            self.layer_type, self.if_batchnorm, self.activation_type))
        
        if self.layer_type == 'Linear':
            print("shape of weight is {}, in format (dim_in, dim_out)".format(self.params_shape))
        else:
            print("shape of weight is {}, in format (c_out, c_in, k_height, k_width)".format(self.params_shape))
        return
    
    def _layer_forward(self, input, mode='train'):
        '''
        cache: list of cache from 'Linear'/'Conv2d', batchnorm and activation
                e.g. [cache_linear, cache_batchnorm, cache_activation]
                
        if some batchnorm or activation function is None, 
        the component cache at the position is also None.
                e.g. [cache_linear, None, cache_activation]
        '''
        cache = []
        
        # Linear/Conv2d
        if self.layer_type == 'Linear':
            out1, cache1 = self._forward_Linear(input)   
        else:
            out1, cache1 = self._forward_Conv2d(input)
        cache.append(cache1)
        
        # batchnorm
        if self.if_batchnorm:
            out2, cache2 = self._forward_batchnorm(out1, mode)
        else:
            out2, cache2 = out1, None
        cache.append(cache2)
        
        # activation
        if self.activation_type == 'tanh':
            output, cache3 = self._forward_tanh(out2)        
        elif self.activation_type == 'relu':
            output, cache3 = self._forward_relu(out2)
        elif self.activation_type == 'sigmoid':
            output, cache3 = self._forward_sigmoid(out2)
        else:
            output, cache3 = out2, None
        cache.append(cache3)
        
        return output, tuple(cache)
            
    def _layer_backward(self, d_output, cache):
        '''
        d_params: dictionary, gradients of weight, bias, gamma and beta
            e.g. {'d_weight', 'd_bias', 'd_gamma', 'd_beta'}
                or {'d_weight', 'd_bias'} if no batchnorm
        '''
        d_params = {}
        
        # activation
        if self.activation_type == 'tanh':
            d_out2 = self._backward_tanh(d_output, cache[2])        
        elif self.activation_type == 'relu':
            d_out2 = self._backward_relu(d_output, cache[2])
        elif self.activation_type == 'sigmoid':
            d_out2 = self._backward_sigmoid(d_output, cache[2])
        else:
            d_out2 = d_output
        
        # batchnorm
        if self.if_batchnorm:
            d_out1, d_bn_params = self._backward_batchnorm(d_out2, cache[1])
        else:
            d_out1, d_bn_params = d_out2, {}
        
        # Linear/Conv2d
        if self.layer_type == 'Linear':
            d_input, d_layer_params = self._backward_Linear(d_out1, cache[0])
        else:
            d_input, d_layer_params = self._backward_Conv2d(d_out1, cache[0])
        
        d_params = {**d_bn_params, **d_layer_params}
        
        return d_input, d_params
    
    def _layer_update_params(self, d_params, learning_rate, regularization):
        
        self.params['weight'] -= learning_rate*d_params['d_weight'] + \
                                    regularization*self.params['weight']
        self.params['bias'] -= learning_rate*d_params['d_bias']
        
        if self.if_batchnorm:
            self.bn_param['gamma'] -= learning_rate*d_params['d_gamma']
            self.bn_param['beta'] -= learning_rate*d_params['d_beta']
            
        return
        
    '''
    Layer forward/backward pass related:
    
        output, cache = _forward_Linear(input, mode='train')
        d_input, d_params = _backward_Linear(d_output, cache)
        
        output, cache = _forward_Conv2d(input, mode='train')
        d_input, d_params = _backward_Linear(d_output, cache)
    '''
    
    def _forward_Linear(self, input):
        # reshape to N*dim_in, in case the input comes from a Conv2d layer
        input_row = input.reshape((input.size(0), -1))

        assert self.params_shape[0] == input_row.size()[1], \
            "_forward_Linear: input and parameter dimension not matched."
        
        # output = Input*W + b
        output = input_row.matmul(self.params['weight']) + self.params['bias']
        
        cache = input
        return output, cache
    
    def _backward_Linear(self, d_output, cache):
        input = cache
        N = input.size(0)
        d_params = {}
        
        # d_input = d_output*(W')
        d_input = d_output.matmul(self.params['weight'].t()).reshape(input.size())
        # d_W = d_input'*d_output
        d_params['d_weight'] = input.reshape((input.size(0), -1)).t().matmul(d_output)
        # d_b = d_output
        d_params['d_bias'] = d_output.sum(dim=0, keepdim=True)
        
        return d_input, d_params
    
    def _forward_Conv2d(self, input):
        assert input.dim()==4, \
            "_forward_Conv2d: this function only accept 4-dimensional input."
        assert self.params['weight'].dim()==4, \
            "_forward_Conv2d: wrong dimension of weight, check constructor and initialization."
        assert input.size(1)==self.params['weight'].size(1), \
            "_forward_Conv2d: channel number unmatching."
        
        # reformulate stride, as it can be stride or (stride_h, stride_w)
        stride = self.conv_param['stride']
        assert (type(stride) is int) or ((type(stride) is tuple) and (len(stride)==2)), \
            "_forward_Conv2d: wrong stride type, should reconstruct the module."
        if type(stride) is int:
            step_h, step_w = stride, stride
        else:
            step_h, step_w = stride
        
        # zero padding
        input_pad = self._zero_padding(input)
        N, C_in, H, W = input_pad.size()
        
        weight, bias = self.params['weight'], self.params['bias']
        C_out, _, K_H, K_W = weight.size()
             
        assert ((H-K_H)%step_h==0) and ((W-K_W)%step_w==0), \
            "_forward_Conv2d: inappropriate kernel size, stride step or padding."
        
        # number of stride along the row and column
        stride_h, stride_w = int((H-K_H)/step_h+1), int((W-K_W)/step_w+1)
        
        output = torch.empty((N, C_out, stride_h, stride_w), dtype=self.dtype, device=self.device)

        for n in range(N):
            for c in range(C_out):
                for h in range(stride_h):
                    for w in range(stride_w):
                        output[n, c, h, w] = (input_pad[n, :, h*step_h:h*step_h+K_H, w*step_w:w*step_w+K_W]*weight[c]).sum()+bias[c]
        
        cache = input_pad
        return output, cache
    
    def _backward_Conv2d(self, d_output, cache):
        
        input_pad = cache
        weight, bias = self.params['weight'], self.params['bias']
        padding, stride = self.conv_param['padding'], self.conv_param['stride']
        if type(stride) is int:
            step_h, step_w = stride, stride
        else:
            step_h, step_w = stride
        
        N, C_in, H, W = input_pad.size()
        C_out, _, K_H, K_W = weight.size()
        _, _, stride_h, stride_w = d_output.size()
        
        d_weight = torch.empty(weight.size(), dtype=self.dtype, device=self.device).fill_(0)
        d_bias = torch.empty(bias.size(), dtype=self.dtype, device=self.device).fill_(0)
        d_input_pad = torch.empty(input_pad.size(), dtype=self.dtype, device=self.device).fill_(0)
        
        # derivation is in the report
        for n in range(N):
            for c in range(C_out):
                for h in range(stride_h):
                    for w in range(stride_w):
                        window = input_pad[n, :, h*step_h:h*step_h+K_H, w*step_w:w*step_w+K_W]
                        d_bias[c] += d_output[n, c, h, w]
                        d_weight[c, :, :, :] += window * d_output[n, c, h, w]
                        d_input_pad[n, :, h*step_h:h*step_h+K_H, w*step_w:w*step_w+K_W] += weight[c, :, :, : ] * d_output[n, c, h, w]
        
        # unpadding
        d_input = self._zero_unpadding(d_input_pad)

        d_params = {'d_weight': d_weight, 'd_bias': d_bias}
        
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
            sample_mean = input.mean(dim=0, keepdim=True)
            sample_var = input.var(dim=0, keepdim=True)
            input_normed = (input - sample_mean)/(sample_var+self.bn_param['eps']).sqrt()
            
            running_mean = momentum*running_mean + (1-momentum)*sample_mean
            running_var = momentum*running_var + (1-momentum)*sample_var
            
            output = gamma*input_normed + beta
            cache = (input, input_normed, sample_mean, sample_var)
        else:
            input_normed = (input - running_mean)/(running_var+self.bn_param['eps']).sqrt()
            output = gamma*input_normed + beta
            cache = None
        
        self.bn_param['running_mean'], self.bn_param['running_var'] = running_mean, running_var
        
        return output, cache
    
    def _backward_batchnorm(self, d_output, cache):
        input, input_normed, sample_mean, sample_var = cache
        gamma, beta = self.bn_param['gamma'], self.bn_param['beta']
        N = input.size(0)
        
        # derivation in the report
        d_input_normed = gamma*d_output
        inv_std = 1.0 / (sample_var+self.bn_param['eps']).sqrt()
        d_sample_var = -0.5*(d_input_normed*(input-sample_mean)).sum(dim=0, keepdim=True)*(inv_std**3)
        d_sample_mean = -1.0*(d_input_normed*inv_std).sum(dim=0, keepdim=True) \
                        -2.0*d_sample_var*(input-sample_mean).mean(dim=0, keepdim=True)
        d_input = d_input_normed*inv_std + 2.0*d_sample_var*(input-sample_mean)/N + 1.0*d_sample_mean/N
        
        d_params = {}
        d_params['d_gamma'] = (d_output*input_normed).sum(dim=0, keepdim=True)
        d_params['d_beta'] = d_output.sum(dim=0, keepdim=True)
        
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
        output = input.sigmoid()
        cache = output
        return output, cache
    
    def _backward_sigmoid(self, d_output, cache):
        assert d_output.size() == cache.size(), \
            "_backward_sigmoid: two inputs should be of same size."
        output = cache
        d_input = d_output*output*(1-output)
        return d_input
    
    def _forward_tanh(self, input):
        output = input.tanh()
        cache = output
        return output, cache
    
    def _backward_tanh(self, d_output, cache):
        assert d_output.size() == cache.size(), \
            "_backward_tanh: two inputs should be of same size."
        output = cache
        d_input = d_output*(1-output.square())
        return d_input
    
    def _forward_relu(self, input):
        output = input.max(torch.empty(input.size(), dtype=self.dtype, device=self.device).fill_(0.0))
        cache = input
        return output, cache
    
    def _backward_relu(self, d_output, cache):
        assert d_output.size() == cache.size(), \
            "_backward_relu: two inputs should be of same size."
        input = cache
        d_input = d_output*(input>0)
        return d_input
    
    '''
    MSE loss related
        loss = _compute_loss(output, target, regularization)
        
        d_loss = _compute_loss_gradient(output, target)
        
        reg_term = _compute_relularization_term(regularization)
    '''
    def _compute_loss(self, output, target, regularization):
        assert output.size() == target.size(), \
            "output and target should be of the same size."
        N = output.size(0)
        loss = (output-target).square().sum()/N
        
        # add l2 regularization
        reg_term = self._compute_regularization_term(regularization)
        loss += reg_term
        
        return loss
    
    def _compute_loss_gradient(self, output, target):
        assert output.size() == target.size(), \
            "output and target should be of the same size."
        # d_loss = 2*(y_hat - y)
        d_loss = 2*(output-target)
        return d_loss
    
    def _compute_regularization_term(self, regularization):
        if self.layer_type == 'Sequential':
            reg_term = 0.0
            for layer in self.layer_sequence:
                reg_term += layer._compute_regularization_term(regularization)
        else:
            reg_term = 0.5*regularization*self.params['weight'].square().sum()
        return reg_term
    
    
    '''
    Parameter initialization related:
    
        _initialize_Linear()
        
        _initialize_Conv2d()
        
        gain = _activation_gain()
    '''
 
    def _initialize_Linear(self):
        # print("self.params_shape is ", self.params_shape, ", format (dim_in, dim_out)")
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
        self.params['bias'] = torch.empty((c_out, ), dtype=self.dtype, device=self.device).normal_(0, std)
        

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
        self.bn_param['gamma'] = \
            torch.empty([1]+list(input.size())[1:], dtype=self.dtype, device=self.device).fill_(1.0)
        self.bn_param['beta'] = \
            torch.empty([1]+list(input.size())[1:], dtype=self.dtype, device=self.device).fill_(0.0)
        self.bn_param['running_mean'] = \
            torch.empty([1]+list(input.size())[1:], dtype=self.dtype, device=self.device).fill_(0.0)
        self.bn_param['running_var'] = \
            torch.empty([1]+list(input.size())[1:], dtype=self.dtype, device=self.device).fill_(0.0)
        self.bn_param['if_initialized'] = True
        return
    
    '''
    Other functional methods:
        input_pad = _zero_padding(input)
        
        input = _zero_unpadding(input_pad)
    '''
             
    def _zero_padding(self, input):
        '''
        Zero padding:
        
        input: 4-dimensional, (num_samples, channel, width, height)
        padding: int or tuple(length=2), e.g. 2, (2, 3)
        
        output: 4-dimensional, (num_samples, channel, width+..., height+...)
        '''
        padding = self.conv_param['padding']
        
        assert input.dim() == 4, \
            '_zero_padding: This function only supports input tensor of 4-dimensional.'
        assert (type(padding) is int) or (type(padding) is tuple), \
            '_zero_padding: Wrong padding params_shape, input type should be int or tuple.'
        
        if type(padding) is int:
            output = torch.empty(input.size(0), input.size(1), input.size(2)+2*padding, input.size(3)+2*padding, dtype=self.dtype, device=self.device).fill_(0)
            output[:, :, padding:padding+input.size(2), padding:padding+input.size(3)] = input
            
        else:
            assert len(padding) == 2, \
                '_zero_padding: Only accept tuple of length 2.'
            output = torch.empty(input.size(0), input.size(1), input.size(2)+2*padding[0], input.size(3)+2*padding[1], dtype=self.dtype, device=self.device).fill_(0)
            output[:, :, padding[0]:padding[0]+input.size(2), padding[1]:padding[1]+input.size(3)] = input
        
        return output.contiguous()
    
    def _zero_unpadding(self, input):
        
        padding = self.conv_param['padding']
        assert input.dim() == 4, \
            '_zero_padding: This function only supports input tensor of 4-dimensional.'
        assert (type(padding) is int) or (type(padding) is tuple), \
            '_zero_padding: Wrong padding params_shape, input type should be int or tuple.'
        
        if type(padding) is int:
            output = input[:, :, padding:-padding, padding:-padding]
            
        else:
            assert len(padding) == 2, \
                '_zero_padding: Only accept tuple of length 2.'
            output = input[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
        return output
    