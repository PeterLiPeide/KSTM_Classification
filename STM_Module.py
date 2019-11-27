#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for Applying Support Tensor Machine Classification with Modified Gaussian Kernel
Work for High-dimensional tensor

Use pytorch backend. CUDA computing supported when GPU is available

Current version still uses numpy backend. cuda version will be supported in the future.


@author: peter
"""

import numpy as np
from numpy import linalg as LA
from tensorly.decomposition import parafac
import tensorly as tl 
# import torch   #When Pytorch is installed

tl.set_backend('numpy')

class rTensor_base():
    """
    Base class for rank-r tensor analysis

    """

    def __init__(self, weight, tdata):
        # tdata should be numpy.ndarray
        self.weight = weight
        self.rank = len(weight)
        self.com_list = tdata
        self.mode = len(tdata)
        self.size = [np.shape(i)[0] for i in tdata]
        

    


class Compressed_Tensor(rTensor_base):
    '''
    Store original tensor and compressed tensors
    Compressed tensor components are stored in a list with size mode * dimension * rank
    '''
    def __init__(self, weight, tdata):
        super().__init__(weight, tdata)
        
    def get_compression(self, proj_matrices):
        # Input includes a list of projection matrices
        # REQUIRE the same matrices for all tensor data
        self.compressed_dim = proj_matrices.P
        self.compression = []
        for j in range(self.mode):
            temp_list = np.zeros((proj_matrices.P[j], self.rank))
            for i in range(self.rank):
                temp_vec = np.zeros(proj_matrices.P[j])
                for l in range(self.rank):
                    for m in range(self.rank):
                        temp_vec += np.dot(proj_matrices.Random_list[j][l], self.com_list[j][:, m])
                temp_list[:, i] = temp_vec / np.sqrt(self.rank)
            self.compression.append(temp_list)
        return None
    

class R_ProjMatrices():
    """ Generate Random Projection Matrix List for High Dimensional Tensor Data
    Input: D -- A list of numbers indicating number of rows in each mode of tensor
           P -- A list of numbers indicating number of rows of the desired projection
           H --  (Rank of Tensor)
    Output: A List of Random Matrices (Multi-dimensional List)
    """
    def __init__(self, D, P, H):
        self.P = tuple(P)
        self.D = tuple(D)
        self.H = H
        self.Random_list = []
        for j in range(len(D)):
            temp = []
            for i in range(H):
                temp_vec = np.random.normal(0, 1, size= P[j] * D[j])
                temp_matrix = np.reshape(temp_vec, (P[j], D[j]))
                temp.append(temp_matrix)
            self.Random_list.append(temp)
            
    def show_mapping(self, a):
        return (self.P[a], self.D[a])
            
    def show_matrix(self, a, b):
        '''
        Show random projection matrix for mode a at rank b
        '''
        return self.Random_list[a][b]



class STM_Classifier():

    def __init__(self, training, label, kernel):
        """
        Construct a support tensor machine classifier.

        Initial input includes training data and self-defined kernel functions

        Note that tensor has to be compressed before 
        """
        self.train = training
        self.label = label
        self.kernel = kernel 
        self.train_size = len(training)
        

    def Kernel_Matrix(self):
        K = np.zeros((self.train_size, self.train_size))
        for i in range(self.train_size):
            for j in range(i + 1):
                K[i, j] = self.kernel(self.train[i], self.train[j])
                K[j, i] = K[i, j]
        self.Kernel_Train = K
        return None

    def ST_identifier(self, beta_hat):
        """
        Require Kernel_Train first
        Identifying support tensors
        """
        indicator = np.zeros(self.train_size)
        for i in range(self.train_size):
            quant = self.label[i] * np.dot(self.Kernel_Train[:, i].T, beta_hat)
            if quant < 1:
                indicator[i] = 1
            else:
                continue
        return indicator

    def model_fit(self, lambda_vec, maxiter = 10000, eta = 1e-4):
        beta_hat = np.ones(self.train_size) * 3
        iteration = 0
        error_est = 1000
        while iteration <= maxiter and error_est >= eta:
            support_vec = self.ST_identifier(beta_hat)
            Multi_matrix = np.diag(support_vec)
            right_matrix = LA.pinv((lambda_vec * self.Kernel_Train + np.dot(self.Kernel_Train, np.dot(Multi_matrix, self.Kernel_Train))))
            beta_new = np.dot(right_matrix, np.dot(self.Kernel_Train, np.dot(Multi_matrix, self.label.T)))
            error_est = LA.norm((beta_new - beta_hat), 1)
            iteration += 1
            beta_hat = beta_new
        self.pred_coefficients = beta_hat
        return None

    def predict(self, new_tdata):
        """
        Predict new tensor data
        new_tdata is either tensor or compressed tensor
        """
        Ki = np.zeros(self.train_size)
        for i in range(self.train_size):
            Ki[i] = self.kernel(new_tdata, self.train[i])
        pred = np.sign(np.dot(Ki, self.pred_coefficients))
        return pred




    
        
        
    

