import torch
import math
from layer_module import Module
from neural_network import NeuralNet
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

print("-------------------------------------------------------")
print("----------- Single layer module (GD)-------------------")
print("-------------------------------------------------------")
# training data
nb=64
data_input = torch.empty((nb, 2), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input-0.5).norm(p=2, dim=1, keepdim=True)<math.sqrt(1/2/math.pi))*1
print("Training data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
print(" ")

mod2 = Module(layer_type='Linear', 
              if_batchnorm=False, 
              activation_type='relu', 
              params_shape=(2, 1), 
              dtype=dtype, device=device)
print("Module information:")
mod2.print_module()
print(" ")

lr = 4e-3
reg = 0.1
loss_arr = []
for i in range(30):
    output, cache = mod2.forward(data_input, mode='train')
    loss, d_loss = mod2.loss(output, data_target, reg)
    loss_arr.append(loss.item())
    d_output, d_params = mod2.backward(d_loss, cache)
    mod2.update_params(d_params, lr, reg)
    
output, _ = mod2.forward(data_input, mode='test')
print("learning_rate = {}, regularization = {}, num_iteration = {}".format(lr, reg, 30))
print(" ")
print("accuracy on training set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")
fig = plt.figure(figsize=(7, 4))
plt.plot(loss_arr)
plt.show()

nb=64
data_input = torch.empty((nb, 2), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input-0.5).norm(p=2, dim=1, keepdim=True)<math.sqrt(1/2/math.pi))*1
print("Testing data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
output, _ = mod2.forward(data_input, mode='test')
print("accuracy on testing set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")

print("-------------------------------------------------------")
print("---------------- End of section -----------------------")
print("-------------------------------------------------------")

input("Press Enter and move on to next section...")

print("-------------------------------------------------------")
print("------------ Sequential module (GD) -------------------")
print("-------------------------------------------------------")
# training data
nb=64
data_input = torch.empty((nb, 2), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input-0.5).norm(p=2, dim=1, keepdim=True)<math.sqrt(1/2/math.pi))*1
print("Training data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
print(" ")

mod3 = Module(layer_type='Sequential', 
              dtype=dtype, device=device)
mod3.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(2, 100), 
              dtype=dtype, device=device))
mod3.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(100, 50), 
              dtype=dtype, device=device))
mod3.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(50, 20), 
              dtype=dtype, device=device))
mod3.append(Module(layer_type='Linear', 
              if_batchnorm=False, 
              activation_type='relu', 
              params_shape=(20, 1), 
              dtype=dtype, device=device))

print("Module information:")
mod3.print_module()
print(" ")

loss_arr = []
for i in range(30):
    output, cache = mod3.forward(data_input, mode='train')
    loss, d_loss = mod3.loss(output, data_target, reg)
    # print("Loss is {:.3f}".format(loss.item()))
    loss_arr.append(loss.item())
    d_input, d_params = mod3.backward(d_loss, cache)
    # print(d_params)
    mod3.update_params(d_params, lr, reg)

print("learning_rate = {}, regularization = {}, num_iteration = {}".format(lr, reg, 30))
print(" ")
output, _ = mod3.forward(data_input, mode='test')
print("accuracy on training set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")
plt.plot(loss_arr)
plt.show()

nb=64
data_input = torch.empty((nb, 2), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input-0.5).norm(p=2, dim=1, keepdim=True)<math.sqrt(1/2/math.pi))*1
print("Testing data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
output, _ = mod3.forward(data_input, mode='test')
print("accuracy on testing set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")

print("-------------------------------------------------------")
print("---------------- End of section -----------------------")
print("-------------------------------------------------------")

input("Press Enter and move on to next section...")

print("-------------------------------------------------------")
print("------------- Sequential module (SGD) -----------------")
print("-------------------------------------------------------")
# training data
nb=1000
data_input = torch.empty((nb, 2), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input-0.5).norm(p=2, dim=1, keepdim=True)<math.sqrt(1/2/math.pi))*1
print("Training data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
print(" ")

print("Sanity check")
fig, ax = plt.subplots(figsize=(5, 5))
circ = plt.Circle((0.5, 0.5), math.sqrt(1/2/math.pi), fill=False)
plt.scatter(data_input[:, 0].cpu(), data_input[:, 1].cpu())
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Scatter of input')
ax.add_patch(circ)
plt.show()
print('Average of train_target==1 is', torch.mean(data_target*1.0).item())
print(" ")

net1 = NeuralNet(layer_type='Sequential', learning_rate=2e-3, 
                 regularization=0.05, epoch=1000, batch_size=64,
                 dtype=dtype, device=device)
net1.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(2, 100), 
              dtype=dtype, device=device))
net1.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(100, 50), 
              dtype=dtype, device=device))
net1.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(50, 20), 
              dtype=dtype, device=device))
net1.append(Module(layer_type='Linear', 
              if_batchnorm=False, 
              activation_type='relu', 
              params_shape=(20, 1), 
              dtype=dtype, device=device))

print("Module information:")
net1.print_module()
print(" ")

print("learning_rate = {}, regularization = {},  ".format(net1.learning_rate, net1.regularization))
print("epoch = {}, batch_size = {}".format(net1.epoch, net1.batch_size))
print(" ")

loss_arr = net1.train(data_input, data_target)
plt.figure(figsize=(7, 4))
plt.plot(loss_arr)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()
output, _ = net1.forward(data_input, mode='test')
print("accuracy on training set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")

test_input = torch.empty((nb, 2), dtype=dtype, device=device).uniform_(0, 1)
test_target = ((test_input-0.5).norm(p=2, dim=1, keepdim=True)<math.sqrt(1/2/math.pi))*1
output, _ = net1.forward(test_input, mode='test')
print("accuracy on testing set is", torch.mean(1.0*((output>0.5)==test_target)).item())
print(" ")


print("-------------------------------------------------------")
print("---------------- End of section -----------------------")
print("-------------------------------------------------------")

input("Press Enter and Exit...")

'''
print("The next session is a neural network with 2 convolutional layers,")
print("It may take quite some time to finish.")
input("Press Enter and move on to next section...")

print("-------------------------------------------------------")
print("--------- Sequential module (SGD+Conv2d) --------------")
print("-------------------------------------------------------")

device = 'cuda:0'
nb=100
data_input = torch.empty((nb, 3, 4, 4), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input.reshape(nb, -1)-0.5).norm(p=2, dim=1, keepdim=True)<2)*1
print("Training data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
print('    Average of train_target==1 is', torch.mean(data_target*1.0).item())
print(" ")

net2 = NeuralNet(layer_type='Sequential', learning_rate=1e-3, 
                 regularization=0.0, epoch=300, batch_size=64,
                 dtype=dtype, device=device)
net2.append(Module(layer_type='Conv2d', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(5, 3, 3, 3), 
              conv_param={'padding':1, 'stride':1},
              dtype=dtype, device=device))
net2.append(Module(layer_type='Conv2d', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(1, 5, 3, 3), 
              conv_param={'padding':1, 'stride':1},
              dtype=dtype, device=device))
net2.append(Module(layer_type='Linear', 
              if_batchnorm=True, 
              activation_type='relu', 
              params_shape=(16, 10), 
              dtype=dtype, device=device))
net2.append(Module(layer_type='Linear', 
              if_batchnorm=False, 
              activation_type='relu', 
              params_shape=(10, 1), 
              dtype=dtype, device=device))

print("Module information:")
net2.print_module()
print(" ")

print("learning_rate = {}, regularization = {},  ".format(net2.learning_rate, net2.regularization))
print("epoch = {}, batch_size = {}".format(net2.epoch, net2.batch_size))
print(" ")

loss_arr = net2.train(data_input, data_target)

plt.figure(figsize=(7, 4))
plt.plot(loss_arr)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()
output, _ = net2.forward(data_input, mode='test')
print("accuracy on training set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")

data_input = torch.empty((nb, 3, 4, 4), dtype=dtype, device=device).uniform_(0, 1)
data_target = ((data_input.reshape(nb, -1)-0.5).norm(p=2, dim=1, keepdim=True)<2)*1
print("Testing data:")
print("    Input size:", data_input.size(), " target size:", data_target.size())
output, _ = net2.forward(data_input, mode='test')
print("accuracy on testing set is", torch.mean(1.0*((output>0.5)==data_target)).item())
print(" ")

print("-------------------------------------------------------")
print("---------------- End of section -----------------------")
print("-------------------------------------------------------")

'''
