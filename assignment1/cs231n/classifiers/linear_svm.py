import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print('dW {}').format(dW.shape)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        #print('dW {}, X[i] {}').format(dW[:, y[i]].shape, X[i].shape)
        loss += margin
        
        dW[:, y[i]] -= X[i]
        dW[:, j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add gradient due to regularization
  dW += (reg * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X, W).T
  correct_class_scores = scores[y, np.arange(num_train)] #contains scores for correct class for all examples
  scores = scores - correct_class_scores + 1  #substract correct class score
  scores[y, np.arange(num_train)] = 0

  # Compute max
  thresh = np.maximum(scores, 0, scores)

  loss = np.sum(thresh)/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  '''
  dW = np.matmul(X.transpose(), thresh) #dW accumulates for other classes over correct class
  #print(np.sum(X[y==0], axis=0))
  X_intermed = X * np.reshape(np.sum(S>0, axis=1), (S.shape[0], -1))

  for i in range(num_classes):
        dW[:,i] -= np.sum(X_intermed[y==i], axis=0)
  '''
  
  # Binarize into integers
  binary = thresh
  binary[thresh > 0] = 1

  # Perform the two operations simultaneously
  # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i].T
  # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i].T
  col_sum = np.sum(binary, axis=0)
  binary[y, range(num_train)] = -col_sum[range(num_train)]
  dW = np.dot(binary, X).T

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add gradient due to regularization
  dW += (reg * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
