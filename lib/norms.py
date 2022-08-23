# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:07:01 2021

@author: osman
"""
import numpy as np

def distance(x,y,norm="L2"):
    if norm is "L1":
        return np.linalg.norm((x.reshape(np.prod(x.shape)) - y.reshape(np.prod(y.shape))), ord=1)
    elif norm is "L2":
        return np.linalg.norm(x.reshape(np.prod(x.shape)) - y.reshape(np.prod(y.shape)))
    elif norm is "inf":
        return np.linalg.norm(x.reshape(np.prod(x.shape)) - y.reshape(np.prod(y.shape)),ord=np.inf)
    elif norm is "ninf":
        return np.linalg.norm(x.reshape(np.prod(x.shape)) - y.reshape(np.prod(y.shape)),ord=-np.inf)
    else:
        print("norm = "+norm+" is not defined.")
            