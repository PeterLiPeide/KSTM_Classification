"""
module of Random projection tensors
Input: Original tensor data, list of random projection matrices
Output: An objection containing random projected data, Frobenius norm of data projection error

Operations: tensor CP decomposition, and random projection


It is a sklearn style model estimation, training, and prediction
Some private methods will be provided in the module 

"""
#%%
import numpy as np
from numpy import linalg as LA 
import tensorly as tl  
from tensorly.decomposition import parafac
from sklearn.metrics import hinge_loss, zero_one_loss
# import multiprocessing as mp     # mp cannot mix with tensorly function since tensorly is already using parallel

"""
Below are global functions that can be used in the model estimation or separately
Basic Functions
"""

def CP_decom(Tensordata, Num_Rank):
    """
    Tensor CP decomposition, return CP components
    Tensordata: a ndarray for tensor
    Num_Rank: Assumed the number of rank (knomn)
    """
    decomposed = parafac(Tensordata, Num_Rank)[1]
    return decomposed

def Get_RP_Matrix(D, P, H):
    """ 
    Generate Random Projection Matrix List for High Dimensional Tensor Data
    Input: D -- A list of numbers indicating original tensor dimension
           P -- A list of numbers indicating projected tensor dimension
           H --  (Rank of Tensor)
    Output: A List of Random Matrices
    """
    Random_Matrix = [[np.random.randn(i, j) for k in range(H)] for (i, j) in zip(P, D)]
    return Random_Matrix

def Project_singleMode(Ori, RPMM):
    """
    Ori: Original Tensor Data
    RPMM: list of random projection matrices for this mode, length = num_of_rank
    """
    R = len(RPMM)
    p = np.shape(RPMM[0])[0]
    temp = np.zeros((p, R))
    for i in RPMM:
        temp += np.dot(i, Ori)
    return temp / np.sqrt(R)

def Get_Projection(Tensor_fac, RPMM_list):
    """
    Applying tensor random projection parallely
    Tensor_fac: tensor factors
    RPMM_list: list of random projection matrices; len = mode, each sublist contains rank many different matrices
    Not doing parallel in projection, but in decomposing & Projecting a list of tensors
    """
    Res = [Project_singleMode(Ori, RPMM) for (Ori, RPMM) in zip(Tensor_fac, RPMM_list)]
    return Res

def Get_KernelValues(Ta, Tb, Kernel_Gamma):
    """
    Ta, Tb: two lists of decomposed (projected) tensor factors
    Kernel_Gamma: Scaling parameters 
    """
    d, R = len(Ta), np.shape(Ta[0])[1]
    total_sum = 0
    for i in range(R):
        for j in range(R):
            prod_total = 1
            for k in range(d):
                diff = Ta[k][:, i] - Tb[k][:, j]
                temp = np.exp(-1 * (1 / Kernel_Gamma[k]) * (LA.norm(diff, 2)**2) 
                / (LA.norm(Ta[k][:, i], 2) * LA.norm(Tb[k][:, j], 2)))
                prod_total *= temp
            total_sum += prod_total
    return total_sum


def Get_KernelMatrix(Tensor_list, Kernel_Gamma):
    """
    A global function to calculate kernel matrix when a list of tensor decompositon list is given
    Tensor_list: a list of decomposed tensor, len = n_training; each element is a decomposed tensor 
    factors, i,e output from fun cp_decom
    Kernel_Gamma: scaling parameter for kernel function
    """
    N = len(Tensor_list)
    ans = np.zeros((N, N))
    K_val = [Get_KernelValues(Tensor_list[i], Tensor_list[j], Kernel_Gamma) for i in range(N) for j in range(i, N)]
    # Turn list to an upper triangular matrix
    Upper_index = np.triu_indices(N)
    ans[Upper_index] = K_val
    Upper_ind_offdiag = np.triu_indices(N, 1)
    Lower_ind_offdiag = np.tril_indices(N, -1)
    ans[Lower_ind_offdiag] = ans[Upper_ind_offdiag]
    return ans




def Get_SV(K, Beta_hat, Ylabel):
    """
    Gloabl function to identify support tensors
    K: Kernal matrix
    Beta_hat: current value of coffecients
    Ylabel: Labels
    """
    Pred = np.dot(K.T, Beta_hat)
    inda = np.multiply(Ylabel, Pred)
    ans = [1 if ind < 1 else 0 for ind in inda]
    return np.array(ans)




def STM_fit(Tx_list, ylabel, lambda_vec, Kernel_Gamma, RPMM_list = None, beta_init = None, \
    maxiter = 10000, eta = 0.05):
    """
    Gloabl function to fit STM classifier. RPSTM will be specificed in the model object
    Input: List of tensor; Y labels; tunning parameters lambda_vec
    Random_list: if not none, then a list of random projection matrices should be provided, and model will do
    RPSTM instead of simple STM
    Output Beta, and training history
    training history is the list of values of objective functions, which is the hing_loss + penalty
    """
    if RPMM_list:
        Tx_list = [Get_Projection(Tensor_fac, RPMM_list) for Tensor_fac in Tx_list]
    N = len(Tx_list)
    beta_hat = 3 * np.ones(N) if not beta_init else beta_init
    K = Get_KernelMatrix(Tx_list, Kernel_Gamma)
    iteration = 0
    error_est = 1000
    Obj_function = []
    err_est_seq = []
    while iteration <= maxiter and error_est >= eta:
        # Get temporary objective function values
        pred_decision = np.dot(K, beta_hat)
        obj_temp = hinge_loss(ylabel, pred_decision) + lambda_vec *  LA.norm(beta_hat, 2)**2
        Obj_function.append(obj_temp)

        # Update beta
        support_vector = Get_SV(K, beta_hat, ylabel)
        MM_matrix = np.diag(support_vector)
        right_mm = LA.pinv((lambda_vec * K + np.dot(K, np.dot(MM_matrix, K))))
        beta_new = np.dot(right_mm, np.dot(K, np.dot(MM_matrix, ylabel.T)))
        error_est = LA.norm((beta_hat - beta_new), 1) / len(beta_hat)      # High dimensional parameter, do not need that strict converge result
        err_est_seq.append(error_est)
        iteration += 1
        beta_hat = beta_new
    return {"Beta" : beta_hat, "History" : {"Objective" : Obj_function, "Error_Est" : err_est_seq}}

"""
Returned information can provide: 1. estimated parameters; 2. sequence of objective function to check if there
is any overfitting; 3. Error sequece to see if the iteration is converge at the end of procedure.

The procedure is optimized with hinge loss, but the final accuracy is calculated with zero_one loss
"""


def STM_pred(New_obs, beta_hat, Tx_list, Kernel_Gamma, RPMM_list = None):
    """
    Making prediction basing on fitted beta_hat.
    New_obs: New Tensor (with decomposition) This can save time when do ensemble
    beta_hat: estimated coefficients
    Tx_list: list of tensor without random projection; If RPMM_list is not none, then Tx_list has 
    to be projected with the same random projection matrices
    Kernel_Gamma: Scaling parameters
    """
    if RPMM_list:
        New_obs = Get_Projection(New_obs, RPMM_list)
        Tx_list = [Get_Projection(tensor, RPMM_list) for tensor in Tx_list]
    Ki = [Get_KernelValues(New_obs, tensor, Kernel_Gamma) for tensor in Tx_list]
    Ki = np.array(Ki)
    pred = np.sign(np.dot(Ki, beta_hat))
    return pred


"""
Ensemble estimation and prediction 
"""

def STM_Ensemble(Tx_list, ylabel, Kernel_Gamma, lambda_vec, RPMM_list_full, beta_init = None, \
    maxiter = 10000, eta = 0.05):
    """
    Trainig STM ensemble; number of ensemble = number of different random projection
    Tx_list: list of Training tensors, decomposed
    ylabel: list of target label of training tensors
    Kernel_Gamma: Scaling parameter in the Kernel function
    lambda_vec: regularization parameter
    RPMM_list_full: list of random projection matrices, length = B (number of ensemble)
    """
    B = len(RPMM_list_full)
    Beta_hat_list = []
    Obj_fun = []
    Error_seq = []
    for i in range(B):
        res = STM_fit(Tx_list, ylabel, lambda_vec, Kernel_Gamma, RPMM_list_full[i], \
            beta_init=beta_init, maxiter = 10000, eta = eta)
        beta_hat, T_history = res["Beta"], res["History"]
        obj, err = T_history["Objective"], T_history["Error_Est"]
        Beta_hat_list.append(beta_hat)
        Obj_fun.append(obj)
        Error_seq.append(err)
    return {"Beta" : Beta_hat_list, "History" : {"Objective" : Obj_fun, "Error_Est" : Error_seq}}


def STM_Ensemble_Pred(New_list, Beta_hat_list, Tx_list, Kernel_Gamma, RPMM_list_full, alpha):
    """
    Making prediction with ensemble classifiers
    New_list: list of new tensors with decomposition
    Beta_hat_list: a list of classifiers' parameters
    Tx_list: training data; Not do random projection, but will do it in the individual prediction function
    Kernel_Gamma: Kernel function
    RPMM_list_full: list of random projection matrices, length = number of ensemble classifiers
    alpha: threshold parameter
    """
    ans = []
    m = len(New_list)
    B = len(RPMM_list_full)
    for i in range(m):
        temp_pred = [STM_pred(New_list[i], Beta_hat_list[j], Tx_list, Kernel_Gamma, \
            RPMM_list_full[j]) for j in range(B)]
        temp_ans = 1 if np.mean(temp_pred) >= alpha else -1
        ans.append(temp_ans)
    return ans



"""
General Idea of Creating Model object:
1. Initial model with key parameters: number of ensemble, rank, P, and D (These information are model specific,
has to be modified if one wants to train / test different models)
   Generate random projection matrix list in the initial function

2. Model Fit: Do tensor decomposition first, and then random projection and train STM
    Include training error in the procedure, also provide objective function's value when optimizing

3. Model Predict: Tensor decomposition -> each random projection and predict STM
    Include testing error if true pred is available 


4. Train_plot: to track training procedure.

"""

import matplotlib.pyplot as plt 

class STM_Classifier():
    """
    Constructing support tensor machine classifier
    """

    def __init__(self, Num_Rank, D = 0, P = 0, num_ensemble = 0, random_projection = True):
        """
        Initial a STM classifier;
        Num_Rank: rank of tensor, is necessary for every STM classifier
        D, P: original and projected tensor dimensions; only necessary when random projection = True
        num_ensemble
        random_projection: if false, do STM without random projection
        """
        self.do_RP = random_projection
        self.rank = Num_Rank
        if not random_projection:
            return
        else:
            self.mode = len(D)
            self.P, self.D = P, D
            self.num_ensemble = num_ensemble
            self.RPMM_list_full = [Get_RP_Matrix(self.D, self.P, self.rank) for i \
                 in range(self.num_ensemble)]
            return
        

    def fit(self, Tx, ylabel, Kernel_Gamma, lambda_vec, beta_init = None, maxiter = 10000, eta = 0.05):
        """
        Fitting STM
        """
        self.Tx_list = [CP_decom(tensor, self.rank) for tensor in Tx]
        self.Kernel_Gamma = Kernel_Gamma
        if not self.do_RP:
            # Not doing random projection, then no ensemble. Just decompose lit and train STM
            temp_res = STM_fit(self.Tx_list, ylabel, lambda_vec, Kernel_Gamma, beta_init=beta_init, maxiter=maxiter,\
                eta=eta)
            self.beta = temp_res["Beta"]
            self.history = temp_res['History']
        else:
            temp_res = STM_Ensemble(self.Tx_list, ylabel, Kernel_Gamma, lambda_vec, self.RPMM_list_full, \
                beta_init=beta_init, maxiter=maxiter, eta=eta)
            self.beta = temp_res['Beta']
            self.history = temp_res['History']
        return 

    def pred(self, newX, alpha = 0):
        new_list = [CP_decom(tensor, self.rank) for tensor in newX]
        if not self.do_RP:
            pred = [STM_pred(newTx, self.beta, self.Tx_list, self.Kernel_Gamma) for newTx in new_list]
        else:
            pred = STM_Ensemble_Pred(new_list, self.beta, self.Tx_list, self.Kernel_Gamma, \
                self.RPMM_list_full, alpha)
        return pred
    
    def pred_acc(self, newX, newY, alpha = 0):
        ans = self.pred(newX, alpha)
        loss = zero_one_loss(ans, newY, normalize=True)
        return 1 - loss

    def Plot_history(self, filename = None):
        """
        plot training procedure to check if there is any overfitting
        """
        if not self.history:
            print("Error! No history data is available.")
            return
        obj_values = self.history['Objective']
        error_est = self.history['Error_Est']
        if not self.do_RP:
            iteration = np.linspace(1, len(obj_values), num=len(obj_values))
            plt.subplot(2, 1, 1)
            plt.plot(iteration, obj_values)
            plt.ylabel('Objective function')
            plt.title('Training History')
            
            plt.subplot(2, 1, 2)
            plt.plot(iteration, error_est)
            plt.ylabel('Parameter gaps between iterations')
            plt.xlabel('Iterations')
            if not filename:
                plt.show()
            else:
                plt.savefig(filename, format='pdf')
            return
        else:
            iteration = np.linspace(1, len(obj_values[0]), num=len(obj_values[0]))
            plt.subplot(2, 1, 1)
            for j in range(len(obj_values)):
                plt.plot(iteration, obj_values[j], label="classifier " + str(j))
            plt.legend()
            plt.ylabel('Objective function')
            plt.title('Training History')
            
            plt.subplot(2, 1, 2)
            for k in range(len(error_est)):
                plt.plot(iteration, error_est[k], label="classifier " + str(k))
            plt.legend()
            plt.ylabel('Parameter gaps between iterations')
            plt.xlabel('Iterations')
            if not filename:
                plt.show()
            else:
                plt.savefig(filename, format='pdf')
            return

    """
    Current methods are all included. Few more will be added when considered
    """
    


    

    



