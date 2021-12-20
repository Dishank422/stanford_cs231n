from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    y_pred = X@W
    num_train = len(X)
    num_classes = W.shape[1]
    for i in range(num_train):
        curr_scores = y_pred[i] #1-D Array of all scores for a particular test sample
        curr_scores = curr_scores-np.max(curr_scores)
        
        softmax = np.exp(curr_scores)/np.sum(np.exp(curr_scores))
        p = softmax[y[i]]
        
        loss += -1*np.log(p)*(1/num_train)
        softmax[y[i]] -= 1
        for j in range(num_classes):
            dW[:,j] += X[i]*softmax[j]*(1/num_train)
            
               
    loss += reg*(np.sum(W*W))
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    y_pred = X@W
    y_pred -= np.max(y_pred, axis=1)[:, np.newaxis]
    num_train = len(X)
    num_classes = W.shape[1]
    
    exp = np.exp(y_pred)
    softmax = exp/np.sum(exp, axis=1)[:,np.newaxis]
    loss = np.sum(-np.log(softmax[np.arange(num_train),y]))*(1/num_train)
    
    softmax[np.arange(num_train),y] -= 1
    dW = X.T@softmax/num_train
    
    loss += reg*(np.sum(W*W))
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
