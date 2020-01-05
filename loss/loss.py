# -*- coding=utf-8 -*-
import numpy as np
from activate.activate import softmax

def mse(y_pred, y_true):
    '''均方差'''
    return np.mean(np.sum(np.square(y_true - y_pred), axis=1)) / 2.0

def mse_gradient(y_pred, y_true):
    return (y_pred - y_true) #/ y_pred.shape[0]


def cross_entropy(y_pred, y_true):
    '''交叉熵'''
    inds = np.argmax(y_true, axis=1)
    y = y_pred[tuple([range(0, y_pred.shape[0]), inds])]
    return np.mean(-np.log(y))

def cross_entropy_gradient(y_pred, y_true):
    d = np.zeros_like(y_pred)
    inds = np.argmax(y_true, axis=1)
    y = y_pred[tuple([range(0, y_pred.shape[0]), inds])]
    y = -1.0 / y
    d[tuple([range(0, y_pred.shape[0]), inds])] = y
    return d / y_pred.shape[0]


def cross_entropy_with_logits(y_pred, y_true):
    '''使用logits求交叉熵'''
    return cross_entropy(softmax(y_pred), y_true)

def cross_entropy_with_logits_gradient(y_pred, y_true):
    return (softmax(y_pred) - y_true) #/ y_pred.shape[0]


class Loss:
    def __init__(self, loss):
        if loss == 'mse':
            self.loss_fn = mse
            self.loss_gradient = mse_gradient
        elif loss == 'cross_entropy':
            self.loss_fn = cross_entropy
            self.loss_gradient = cross_entropy_gradient
        elif loss == 'cross_entropy_with_logits':
            self.loss_fn = cross_entropy_with_logits
            self.loss_gradient = cross_entropy_with_logits_gradient
        else:
            pass
        self.epson = 1e-7

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred + self.epson
        self.y_true = y_true
        return self.loss_fn(self.y_pred, y_true)

    def backward(self):
        return self.loss_gradient(self.y_pred, self.y_true)
