from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from scipy.spatial import distance_matrix

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def rbf(x_1, x_2, sigma=1.):
    '''Computes rbf kernel for batches of objects

    Args:
        x_1: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
        x_2: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
    Returns:
        kernel function values for all pairs of samples from x_1 and x_2
        torch.tensor of type torch.float32 shaped `(#samples_1, #samples_2)`
    '''  
    ### YOUR CODE HERE
    # The first shortest variant using scipy
    # distances = distance_matrix(x_1, x_2)
    
    # The second variant using numpy/pytorch broadcasting
    pairs = torch.unsqueeze(x_1, dim=1) - x_2
    distances = np.sqrt(torch.sum((pairs)**2, 2) 
                        
    # The third longest variant
    # x_1 = x_1.numpy()
    # x_2 = x_2.numpy()
    # samples_1 = x_1.shape[0]
    # samples_2 = x_2.shape[0]
    # x_1_dots = (x_1*x_1).sum(axis=1).reshape((samples_1,1))*np.ones(shape=(1,samples_2))
    # x_2_dots = (x_2*x_2).sum(axis=1)*np.ones(shape=(samples_1, 1))
    # x_1_x_2 = -2*x_1.dot(x_2.T)
    # distances = torch.from_numpy(np.sqrt(x_1_dots + x_2_dots + x_1_x_2))
    
    distances = torch.exp(-torch.tensor(distances).type(torch.float32) / (2*sigma**2))
   
    return torch.Tensor(distances).type(torch.float32)

def hinge_loss(scores, labels):
    '''Mean loss for batch of objects
    '''
    assert len(scores.shape) == 1
    assert len(labels.shape) == 1
    return torch.clamp(1 - scores * labels, min=0).mean() ### YOUR CODE HERE


class SVM(BaseEstimator, ClassifierMixin):
    @staticmethod
    def linear(x_1, x_2):
        '''Computes linear kernel for batches of objects
        
        Args:
            x_1: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
            x_2: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
        Returns:
            kernel function values for all pairs of samples from x_1 and x_2
            torch.tensor shaped `(#samples_1, #samples_2)` of type torch.float32
        '''
        
        return x_1 @ x_2.T ### YOUR CODE HERE
    
    def __init__(
        self,
        lr: float=1e-3,
        epochs: int=2,
        batch_size: int=64,
        lmbd: float=1e-4,
        kernel_function=None,
        verbose: bool=False,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lmbd = lmbd
        self.kernel_function = kernel_function or SVM.linear
        self.verbose = verbose
        self.fitted = False

    def __repr__(self):
        return 'SVM model, fitted: {self.fitted}'

    def fit(self, X, Y):
        assert (np.abs(Y) == 1).all()
        n_obj = len(X)
        X, Y = torch.FloatTensor(X), torch.FloatTensor(Y) # X.shape = (100, 2), Y.shape = (100, )
        K = self.kernel_function(X, X).float() # K.shape = (100, 100)

        self.betas = torch.full((n_obj, 1), fill_value=0.001, dtype=X.dtype, requires_grad=True) # betas.shape = (100, 1)
        self.bias = torch.zeros(1, requires_grad=True) # I've also add bias to the model self.bias.shape = (1, )
        
        optimizer = optim.SGD((self.betas, self.bias), lr=self.lr)
        for epoch in range(self.epochs):
            perm = torch.randperm(n_obj)  # Generate a set of random numbers of length: sample size
            sum_loss = 0.                 # Loss for each epoch
            for i in range(0, n_obj, self.batch_size):
                batch_inds = perm[i:i + self.batch_size]
                x_batch = X[batch_inds]   # Pick random samples by iterating over random permutation x_batch.shape = (20, 2)
                y_batch = Y[batch_inds]   # Pick the correlating class y_batch.shape = (20, )
                k_batch = K[batch_inds] # k_batch.shape = (20, 100)
                
                optimizer.zero_grad()     # Manually zero the gradient buffers of the optimizer
                
                ### YOUR CODE HERE # get the matrix product using SVM parameters: self.betas and self.bias
                preds = k_batch @ self.betas + self.bias
                
                preds = preds.flatten() # preds.shape = (20, )
                # (1, 20) @ (20, 100) @ (100, 1) + hinge_loss((20, ),(20, )) = ()
                loss = (self.lmbd * self.betas[batch_inds].T @ k_batch @ self.betas) + hinge_loss(preds, y_batch)
                loss.backward()           # Backpropagation
                optimizer.step()          # Optimize and adjust weights

                sum_loss += loss.item()   # Add the loss

            if self.verbose: print("Epoch " + str(epoch) + ", Loss: " + str(sum_loss / self.batch_size))

        self.X = X
        self.fitted = True
        return self

    def predict_scores(self, batch):
        with torch.no_grad():
            batch = torch.from_numpy(batch).float()
            K = self.kernel_function(batch, self.X)
            # compute the margin values for every object in the batch
            return (K @ self.betas  + self.bias).flatten() ### YOUR CODE HERE (1, 100)@(100,20) - (1, ) = (20, )

    def predict(self, batch):
        scores = self.predict_scores(batch)
        answers = np.full(len(batch), -1, dtype=np.int64)
        answers[scores > 0] = 1
        return answers