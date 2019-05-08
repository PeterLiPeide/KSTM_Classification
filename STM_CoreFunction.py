"""
This includes all core function for applying
Kernelized Support Tensor Machine Model
for all data set

A moduler version is under development

An example from our simulation study is given for demostration.

Version 1.0

Author: Peide Li, Taps Maiti
"""

import numpy as np
from numpy import linalg as LA
from tensorly.decomposition import parafac
import imageio

def CP_decom(Tensordata, Num_Rank):
   decomposed = parafac(Tensordata, Num_Rank)
   return decomposed


def Decomposed_List(Training_Tensor, Num_Rank):
    Decomposed = {}
    for i in range(len(Training_Tensor)):
        Decomposed[i] = CP_decom(Training_Tensor[i], Num_Rank)
    return Decomposed

def Kernel_value(A_decomposed, B_decomposed, Num_Rank, Kernel_Gamma):
    num_modes = len(A_decomposed)
    total_sum = 0
    for i in range(Num_Rank):
        for j in range(Num_Rank):
            prod_total = 1
            for k in range(num_modes):
                diff = A_decomposed[k][:, i] - B_decomposed[k][:, j]
                temp = np.exp(-1 * (1 / Kernel_Gamma[j]) * LA.norm(diff, 2) * LA.norm(diff, 2) / (LA.norm(A_decomposed[k][:, i], 2) * LA.norm(B_decomposed[k][:, j], 2))) 
                prod_total *= temp
            total_sum += prod_total
    return total_sum

#    Getting Kernel Matrix
def Kernel_Matrix(Training_Tensor, Num_Rank, Kernel_Gamma):
    d = len(Training_Tensor)
    K = np.zeros((d, d))
    train_list = Decomposed_List(Training_Tensor, Num_Rank)
    for i in range(d):
        for j in range(i + 1):
                K[i, j] = Kernel_value(train_list[i], train_list[j], Num_Rank, Kernel_Gamma)
    for i in range(d):
        for j in range(i + 1, d):
                K[i, j] = K[j, i]
    return K, train_list

#    Idenfying support Tensors
def SupportTensor(K, Training_Label, beta_hat):
    d = len(Training_Label)
    indicator = np.zeros(d)
    for i in range(d):
        quant = Training_Label[i] * np.dot(K[:, i].T, beta_hat)
        if quant < 1:
            indicator[i] = 1
        else: 
            continue
    return indicator
    
#    Fitting STM with SGD
def Model_fit(Training_Tensor, Training_Label, Num_Rank, Kernel_Gamma,lambda_vec, maxiter = 10000, eta = 0.0001):
    d = len(Training_Label)
    Kernel_info = Kernel_Matrix(Training_Tensor, Num_Rank, Kernel_Gamma)
    K = Kernel_info[0]
#    K = Naive_Kernel_Matrix(Training_Tensor, Kernel_Gamma)
    beta_hat = 3 * np.ones(d)
    iteration = 0
    error_est = 100000
    while iteration <= maxiter and error_est >= eta:
        support_vector = SupportTensor(K, Training_Label, beta_hat)
        Multi_Matrix = np.diag(support_vector)
        right_matrix = LA.pinv((lambda_vec * K + np.dot(K, np.dot(Multi_Matrix, K))))
        beta_new = np.dot(right_matrix, np.dot(K, np.dot(Multi_Matrix, Training_Label.T)))
        error_est = LA.norm((beta_new - beta_hat), 1)
        iteration += 1
        beta_hat = beta_new
    return beta_hat, Kernel_info[1]

    
    
#    Prediction Function
def Model_Predict(NewObs, beta_hat, dlist, Num_Rank, Kernel_Gamma):
    Ki = np.zeros(len(dlist))
    Newdec = CP_decom(NewObs, Num_Rank)
    for i in range(len(Ki)):
        Ki[i] = Kernel_value(dlist[i], Newdec, Num_Rank, Kernel_Gamma)
    pred = np.sign(np.dot(Ki, beta_hat))
    return pred
#
