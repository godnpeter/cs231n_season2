import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_class = W.shape[1]
  
  scores = np.matmul(X, W) #(N, C)
  
  scores -= np.reshape(np.max(scores, axis=1), (num_train, 1))
  loss = 0
  
  for i in range(num_train):
    true_class = y[i]
    total_sum = 0
    for j in range(num_class):
      total_sum += np.exp(scores[i,j])
    
    for j in range(num_class):
      dW[:,j] += X[i,:]*(np.exp(scores[i,j])/total_sum)
    dW[:,y[i]] -= X[i]
    
    true_score = np.exp(scores[i,y[i]])
    loss += -np.log(true_score/total_sum)
  
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dim = X.shape[1]
  num_class = W.shape[1]
  
  scores = np.matmul(X, W) #(N, C)
  
  stable_scores = scores - np.reshape(np.max(scores, axis=1), (num_train, 1))
  
  exp_scores = np.exp(stable_scores)
  
  
  exp_scores_sum = np.reshape(np.sum(exp_scores, axis=1), (num_train, 1))
  
  softmax = exp_scores/exp_scores_sum
  
  loss = np.sum(-np.log(softmax[np.arange(num_train), y]))
  
   #brilliant idea... really
  softmax[np.arange(num_train), y] -= 1
  
  dW = np.matmul(X.T, softmax)
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

