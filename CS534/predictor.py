#-------------------------------------------------------------------------------
# Name:        predictor
# Purpose:
#
# Author:      Shi
#
# Created:     31/03/2016
# Copyright:   (c) Administrator 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import kernel

MIN_SUPPORT_VECTOR_MULTIPLIER=0.01
#path="multipliers/"
#path="features_all_without_skeleton/"
#path="features_all_compress5/"

class predictor():
    def __init__(self, kernel, id,path):
        self._bias=np.load(path+id+'.npz')['bias']
        self._support_multipliers = np.load(path+id+'.npz')['support_multipliers']
        self._support_vectors = np.load(path+id+'.npz')['support_vectors']
        self._support_vector_labels = np.load(path+id+'.npz')['support_vector_labels']
        self._kernel=kernel
    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._support_multipliers,
            self._support_vectors,
            self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()

def predictor_new(kernel, classifier, x):

        result = classifier.bias

        for z_i, x_i, y_i in zip(classifier.support_vectors_multipliers,
                                 classifier.support_vectors,
                                 classifier.support_vector_labels):
            result += z_i * y_i * kernel(x_i, x)
        return np.sign(result).item()