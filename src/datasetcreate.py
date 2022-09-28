import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../lib/')

import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
import time
import argparse

import conjugate_gradient as cg
#import pressure_laplacian as pl
import helper_functions as hf

#this makes sure that we are on cpu
os.environ["CUDA_VISIBLE_DEVICES"]= ''