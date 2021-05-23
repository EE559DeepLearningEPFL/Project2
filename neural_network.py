import torch
from layer_module import Module

class NeuralNet(Module):
    
    def __init__(self,
                 layer_type,
                 if_batchnorm=None, 
                 activation_type=None, 
                 params_shape=None, 
                 bn_param={'eps':1e-5, 'momentum':0.9}, 
                 conv_param={'padding':0, 'stride':1},
                 learning_rate=1e-3,
                 regularization=0.0,
                 iteration=100,
                 batch_size=32,
                 dtype=torch.float32, 
                 device=None):

        super().__init__(layer_type, 
                         if_batchnorm,
                         activation_type,
                         params_shape,
                         bn_param,
                         conv_param,
                         dtype,
                         device)
        
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iteration = iteration
        self.batch_size = batch_size
        
    def train(self, input, target):
        lr = self.learning_rate/self.batch_size
        reg = self.regularization/self.batch_size
        
        loss_arr = []
        for e in range(self.iteration):
            indices = torch.empty((self.batch_size, ), 
                                  dtype=torch.int64, 
                                  device=self.device).random_(0, input.size(0))
            
            input_batch = input[indices]
            target_batch = target[indices]
            
            output, cache = self.forward(input_batch, mode='train')
            loss, d_loss = self.loss(output, target_batch, reg)
            loss_arr.append(loss.item())
            d_input, d_params = self.backward(d_loss, cache)
            self.update_params(d_params, lr, reg)
            
        return loss_arr