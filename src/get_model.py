import sys, os
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(1, dir_path+'/../lib/')
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf 
#import scipy.sparse as sparse


class get_model:
    def __init__(self,N):
        self.dim = 3 #       
        self.N = N    
    
    #Todo: Write write_phi_grid function
    def get_predefined_model(self, name_model):
        if name_model == "V101":
            # get that from library, get model
            fil_num=16
            input_rhs = keras.Input(shape=(self.N, self.N, self.N, 1))
            first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
            la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
            lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
            la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
            lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

            apa = layers.AveragePooling3D((2, 2,2), padding='same')(lb) 
            apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
            apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
            apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
            apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
            apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
            apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa

            upa = layers.UpSampling3D((2, 2,2))(apa) + lb
            upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
            upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa
            upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
            upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

            last_layer = layers.Dense(1, activation='linear')(upa)

            model = keras.Model(input_rhs, last_layer)
            return model