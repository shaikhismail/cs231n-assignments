import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = np.matmul(X[i], W)
    scores -= np.amax(scores)
    scores = np.reshape(scores, (scores.shape[0], -1))
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores)
    p = exp_scores/exp_scores_sum
    loss -= np.log(p[y[i]])
    
    grad_term = p * X[i]
    dW += grad_term.transpose()
    dW[:, y[i]] -= X[i]
    
    '''
    #This also works, but uses loops
    for j in range(num_classes):
        if j == y[i]:
            dW[:, y[i]] -= X[i]
            dW[:, y[i]] += p[y[i]] * X[i]
        else:
            dW[:, j] += p[j] * X[i]
    '''
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add gradient due to regularization
  dW += (reg * W)
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = np.matmul(X, W)
  S -= np.amax(S)
  exp_S = np.exp(S)
  exp_S_row_sum = np.sum(exp_S, axis=1, keepdims=True)
  #exp_S_row_sum = np.reshape(exp_S_row_sum, (exp_S_row_sum.shape[0], -1))
  probs = exp_S/exp_S_row_sum

  loss -= np.sum(np.log(probs[range(num_train), y]))
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
    
  #grad
  dS = probs
  dS[range(num_train), y] -= 1
  dW = np.dot(X.T, dS)
  dW /= num_train
  # Add gradient due to regularization
  dW += (reg * W)

  '''
  for j in range(num_classes):
        grad_term = np.reshape(P[:, j], (num_train, -1)) * X
        grad_term = np.sum(grad_term, axis=0)
        dW[:, j] += grad_term.transpose()
        dW[:, j] -= np.sum(X[y==j], axis=0)
  '''
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  
  return loss, dW
  '''
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.        #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  prob_scores = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  correct_log_probs = -np.log(prob_scores[range(num_train), y])
  loss = np.sum(correct_log_probs)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  # grads
  dscores = prob_scores
  dscores[range(num_train), y] -= 1
  dW = np.dot(X.T, dscores)
  dW /= num_train
  dW += reg * W

  return loss, dW
  '''
