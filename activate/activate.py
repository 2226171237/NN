# -*- coding=utf-8 -*-
import numpy as np

def relu(x):
    return np.maximum(x,0)

def relu_gradient(a):
    x=np.ones_like(a)
    x[a<=0]=0
    return x

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_gradient(a):
    return a*(1-a)

def tanh(x):
    return 2*sigmoid(2*x)-1

def tanh_gradient(a):
    return 1-a**2

def linear(x):
    return x

def linear_gradient(a):
    return np.ones_like(a)

def softmax(x):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=-1),axis=-1)

def softmax_gradient(a):
    return

##########################
def activation_fn(name=None):
    if name==None:
        return linear
    elif name=='relu':
        return relu
    elif name=='sigmoid':
        return sigmoid
    elif name=='tanh':
        return tanh
    else:
        raise ValueError('non this activation function')

def activation_gradient(name=None):
    if name==None:
        return linear_gradient
    elif name=='relu':
        return relu_gradient
    elif name=='sigmoid':
        return sigmoid_gradient
    elif name=='tanh':
        return tanh_gradient
    else:
        raise ValueError('non this activation function')

if __name__ == '__main__':
    x=np.array([[1,3,2,4],[2,4,5,6]])
    print(softmax(x))