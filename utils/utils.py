# -*- coding=utf-8 -*-
import numpy as np

def one_hot(y, depth):
    if not isinstance(y, (list, tuple,np.ndarray)):
        y = [y]
    labels = np.zeros(shape=(len(y), depth))
    labels[(range(len(y)), y)] = 1.0
    return labels

