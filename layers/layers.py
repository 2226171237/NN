# -*- coding=utf-8 -*-
import numpy as np
from activate.activate import activation_fn,activation_gradient


class Layer:
    '''全连接层'''
    def __init__(self, input_shape, units, activation=None, trainable=True):
        self.input_shape = input_shape
        self.units = units
        self.activation = activation_fn(activation)
        self.activation_gradfn = activation_gradient(activation)
        self._trainable = trainable  # 是否训练
        self.trainabel_variables = []  # 可训练的变量
        self.non_trainable_variables = []  # 不可训练变量
        self.variables = []  # 所有变量
        self.error = None  # 传给下一层的dL/da
        self.delta = None  # dL/dz
        self.last_output = None  # 最终经过激活函数的输出

        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 2:
                self.weight_shape = (input_shape[1], units)
            else:
                self.weight_shape = (input_shape[0], units)
        else:
            self.weight_shape = (input_shape, units)
        self.build()

    def build(self):
        '''建立参数'''
        self.weights = np.random.normal(loc=0., scale=0.1, size=self.weight_shape).astype(np.float32)
        self.bias = np.zeros(shape=self.units, dtype=np.float32)
        self.variables = [self.weights, self.bias]
        if self._trainable:
            self.trainabel_variables = [self.weights, self.bias]
        else:
            self.non_trainable_variables = [self.weights, self.bias]

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, flag):
        self._trainable = flag
        if self._trainable:
            self.trainabel_variables = self.variables
        else:
            self.non_trainable_variables = self.variables

    def __call__(self, x):
        self.input = x
        z = np.matmul(x, self.weights) + self.bias
        self.last_output = self.activation(z)
        return self.last_output

    def backward(self, error):
        '''反向传播误差'''
        self.delta = self.activation_gradfn(self.last_output) * error
        self.error = np.matmul(self.delta, self.weights.T)

    def gradient(self):
        '''根据反向传播的误差，求导'''
        weight_grad = np.expand_dims(self.delta, axis=1) * np.expand_dims(self.input, axis=-1)
        weight_grad = np.mean(weight_grad, axis=0)
        bias_grad = np.mean(np.sum(self.delta, axis=1))
        return [weight_grad, bias_grad]
