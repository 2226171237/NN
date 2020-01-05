# -*- coding=utf-8 -*-

def sgd(vars,grads,lr):
    if isinstance(vars,(list,tuple)):
        for var,grad in zip(vars,grads):
            var-=lr*grad
    else:
        vars-=lr*grads

class Optimizers:
    def __init__(self,optimizer,learning_rate):
        if optimizer=='sgd':
            self.update_method=sgd
        self.learning_rate=learning_rate

    def update(self,vars,grads):
        self.update_method(vars,grads,self.learning_rate)
