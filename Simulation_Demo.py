"""
This is one of our simulation studies.
One can see how to apply our model to Classification applications

"""
import time
import numpy as np
from numpy import linalg as LA
from tensorly.decomposition import parafac


np.random.seed(9001)

            
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
#    

    
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
    error_est = 1000
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
    
# Creating AR matrix
def AR_Matrix(rho, n):
    cov = np.arange(n*n).reshape(n, n)
    for i in range(n):
        for j in range(n):
            cov[i, j] = rho ** np.abs(i - j)
    return cov

#    
cov3 = np.arange(900).reshape(30, 30)
for i in range(30):
    for j in range(30):
        cov3[i,j] = min(i + 1, j + 1)
#
cov1 = np.diag(np.ones(30))
cov2 = AR_Matrix(0.7, 30)


# Generate variables
def Generate_SmallNormal_Tensor(mean, Num_rank, cov1, cov2, cov3):
    total = 0
    for i in range(Num_rank):
        mode1 = np.random.multivariate_normal(mean, cov1)
        mdoe2 = np.random.multivariate_normal(mean, cov2)
        mode3 = np.random.multivariate_normal(mean, cov3)
        temp = np.tensordot(mode1, mdoe2, axes=0)
        temp = np.tensordot(temp, mode3, axes=0)
        total += temp
    return total

def Generate_Random_Tensor(Num_Rank, unibound, mean, cov2, cov3):
    temp = 0
    for i in range(Num_Rank):
        mode1 = np.random.uniform(unibound[0], unibound[1], 30)
        mode2 =  np.random.multivariate_normal(mean, cov2)
        mode3 = np.random.multivariate_normal(mean, cov3)
        ten = np.tensordot(mode1, mode2, axes = 0)
        ten = np.tensordot(ten, mode3, axes = 0)
        temp += ten
    return temp


#
# One can choose to generate random uniform tensor or normal tensor
data_size = 100
Num_Rank = 3
Class_data = {}
for i in range(data_size):
#    Class_data[i] = Generate_SmallNormal_Tensor(np.zeros(30), 3, cov1, cov2, cov3)
#    Class_data[i + data_size] = Generate_SmallNormal_Tensor(1 * np.ones(30), 3, cov1, cov2, cov3)
    Class_data[i] = Generate_Random_Tensor(3, np.array([0, 1]), np.zeros(30), cov2, cov3)
    Class_data[i + data_size] = Generate_Random_Tensor(3, np.array([1, 2]), np.ones(30), cov2, cov3)
#
    

Kernel_Gamma = np.array([1, 1, 1])

data_label = np.hstack((np.ones(data_size), -1*np.ones(data_size)))


start = time.time()
acc = Model_fit(Class_data, data_label, Num_Rank, Kernel_Gamma, 1)
end = time.time()
print("The training time for STM is %f \n" % (end - start))


# Large testing set
STMerr = 0
n_test = 1000
for j in range(n_test):
#    New_data1 = Generate_SmallNormal_Tensor(np.zeros(30), 3, cov1, cov2, cov3)
    New_data1 = Generate_Random_Tensor(3, np.array([0, 1]), np.zeros(30), cov2, cov3)
    predict1 = Model_Predict(New_data1, acc[0], acc[1], Num_Rank, Kernel_Gamma)
    if predict1 != 1:
        STMerr += 1
#    predict1 = Model_Predict(New_data1, acc, Class_data, Kernel_Gamma)
#    New_data2 = Generate_SmallNormal_Tensor(1 * np.ones(30), 3, cov1, cov2, cov3)
    New_data2 = Generate_Random_Tensor(3, np.array([1, 2]), np.ones(30), cov2, cov3)
#    predict2 = Model_Predict(New_data2, acc, Class_data, Kernel_Gamma)
    predict2 = Model_Predict(New_data2, acc[0], acc[1], Num_Rank, Kernel_Gamma)
    print("The STM predictions are %d and %d \n" %(predict1, predict2))
    if predict2 != -1:
        STMerr += 1
        
        
STMrate = STMerr/(n_test * 2)


print("STM error rate is %f \n" % (STMrate))










