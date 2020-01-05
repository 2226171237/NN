# -*- coding=utf-8 -*-

class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def __call__(self, X):
        for layer in self._layers:
            X = layer(X)
        return X

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, X, y):
        y_pred = self(X)
        loss = self.loss(y_pred, y)
        self.backward(self.loss.backward())
        return loss

    def backward(self, loss_error):
        # 反向传播
        for i, layer in enumerate(self._layers[::-1]):
            if i == 0:
                layer.backward(loss_error)
            else:
                layer.backward(self._layers[len(self._layers) - i].error)
        # 求导
        for layer in self._layers:
            grads = layer.gradient()
            # 更新
            self.optimizer.update(layer.trainabel_variables, grads)
