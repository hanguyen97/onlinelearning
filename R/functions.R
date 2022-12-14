#' Sample i.i.d data
#'
#' @param reg sample data from either `reg="linear"` or `reg = "logistic"`
#' @param n sample size
#' @param b_len length of coefficient vector
#' @param bn0_len length of non-zero coefs
#' @param signal standard deviation of an independent Gaussian distribution with mean 0
#' from which the non-zero coefs are drawn from
#' @param noise standard deviation of a normal distribution with mean 0
#' from which the error vector is drawn from
#' @importFrom stats rbinom rnorm
#' @returns
#'* `y` response vector
#'* `X` signal data
#'* `b` coefficient vector
#'* `e` residual (for linear regression only)
#'* `py` probability vector P(y_t = 1| x_t) (for logistic regression only)
#' @export
#' @examples 
#' set.seed(123)
#' data_lr <- sample_iid_data(reg="linear", n=1000, b_len=10, bn0_len=8)
#' data_lg <- sample_iid_data(reg="logistic", n=1000, b_len=10, bn0_len=8)
sample_iid_data <- function(reg, n=1000, b_len=500, bn0_len=100, signal=5, noise=1) {

  # generate X, beta and error
  X = matrix(rnorm(n*b_len), nrow = n, ncol = b_len)
  b = c(rnorm(bn0_len, mean=0, sd=signal), rep(0, b_len - bn0_len))
  # Notes: add 0s to show that the underlying data structure is simple.
  # Can we do better than the average regret rate?
  e = rnorm(n, 0, noise) # for linear reg only

  # generate response
  index = sample(1:n)
  if (reg=="linear") {
    y = X[, 1:bn0_len] %*% b[1:bn0_len] + e
    return(list(y=y[index], X=X[index,], b = b, e = e[index]))

  } else if (reg=="logistic") {
    py = exp(X[, 1:bn0_len] %*% b[1:bn0_len])/(1+exp(X[, 1:bn0_len] %*% b[1:bn0_len]))
    y = rbinom(n, 1, py)
    return(list(y=y[index], X=X[index,], b = b, py=py[index]))
  }
}


#' Run recursive least squares
#'
#' @param Y response vector
#' @param X design matrix
#' @param initial how many data points you have used in your previous model
#' @param invxtx \eqn{(X^{T}X)^{-1}} from previous model. If there is no previous OLS model 
#' \eqn{10^{6}I_{p}} is used where p is the number of regression parameters
#' @param xty matrix \eqn{X^{T}y} from previous OLS model. If there is no previous model
#' \eqn{0_{p}} is used.
#' @param btrue true coefficients
#'
#' @return
#'* `beta` estimated coefficients
#'* `eeps` estimation error
#'* `peps` prediction error
#' @export
#' @examples 
#' testdata = c(1,0,0,1,1,1,1,2,3,1,3,5,1,4,4,1,5,4)
#' testdata = matrix(testdata,nrow = 6,ncol=3,byrow = TRUE)
#' response = c(3,5,6,8,10,11)
#' (beta = run_RLS(response,testdata)[1])
run_RLS = function(Y,X, initial=0,invxtx= 10^6 *diag(nrow = dim(X)[2],ncol = dim(X)[2]),xty = matrix(0,nrow = dim(X)[2],ncol = 1),btrue = rep(0,dim(X)[2])){

  beta = invxtx %*% xty
  eeps = rep(1,dim(X)[2])
  peps = rep(0,dim(X)[2])
  for(i in 1:(length(Y)-initial)){
    beta = beta + as.double((1/(1+as.matrix(t(X[initial+i,])) %*% invxtx %*% as.matrix(X[initial+i,])))) * invxtx %*% as.matrix(X[initial+i,]) %*% (Y[initial +i] - t(as.matrix(X[initial+i,])) %*% beta)
    newx = as.matrix(X[initial+i,]) %*% t(as.matrix(X[initial+i,])) %*% invxtx
    g = sum(diag(newx))
    invxtx = invxtx - 1/(1+g) *invxtx %*% newx
    eeps[i] = norm(beta-btrue,type = "2")
    peps[i] = norm(Y - X %*% beta, type="2")/nrow(X)
  }
  return(list(beta,eeps,peps))
}


#' Run online gradient descent
#'
#' @param reg run linear `reg="linear"` or logistic `reg = "logistic"` regression
#' @param y response vector
#' @param X signal data
#' @param b_true true coefficients
#' @param learn_rate learning rate
#' @param PR_ave logical; for Polyak-Ruppert averaging
#'
#' @return
#'* `b_est` estimated coefficients
#'* `b_true` true coefficients
#'* `est_error` estimation error
#'* `pred_error` prediction error
#' @export
#' @examples 
#' set.seed(123)
#' data <- sample_iid_data(reg="linear", n=1000, b_len=10, bn0_len=8)
#' run_OGD(y=data$y, X=data$X, b_true=data$b, reg="linear", learn_rate=0.01)
run_OGD <- function(reg, y, X, b_true, learn_rate, PR_ave=TRUE) {

  b_len = length(b_true)
  n_iter = nrow(X)

  # initialize beta
  b_est = numeric(b_len)
  b_est_matrix = matrix(0, nrow = b_len, ncol = n_iter)
  est_error = numeric(n_iter)
  pred_error = numeric(n_iter)

  # loop through the most recent data point
  for (i in 1:n_iter) {

    # Update the gradient
    gradient = get_gradient(x=X[i,], y[i], b=b_est, reg=reg)

    # Update beta
    b_est_raw = b_est - learn_rate/sqrt(i) * gradient
    if (PR_ave==TRUE) {
      # Polyak-Ruppert Averaging
      b_est = (b_est_raw + b_est*(i-1))/i
    } else {
      b_est = b_est_raw
    }
    # Store new beta
    b_est_matrix[, i] = b_est

    # Calculate estimation and prediction errors
    est_error[i] = est_error(b_true=b_true, b_est=b_est)
    pred_error[i] = pred_error(reg=reg, y=y, X=X, b_est=b_est)
  }

  return(list(b_est = b_est,
              # b_est_past = b_est_matrix,
              b_true = b_true,
              est_error = est_error,
              pred_error = pred_error))
}


#' Run adaptive gradient descent
#'
#' @param reg run linear `reg="linear"` or logistic `reg = "logistic"` regression
#' @param y response vector
#' @param X signal data
#' @param b_true true coefficients
#' @param learn_rate learning rate
#' @param PR_ave logical; for Polyak-Ruppert averaging
#'
#' @return
#'* `b_est` estimated coefficients
#'* `b_true` true coefficients
#'* `est_error` estimation error
#'* `pred_error` prediction error
#' @export
#' @examples 
#' set.seed(123)
#' data <- sample_iid_data(reg="linear", n=1000, b_len=10, bn0_len=8)
#' run_AdaGrad(y=data$y, X=data$X, b_true=data$b, reg="linear", learn_rate=0.01)
run_AdaGrad <- function(reg, y, X, b_true, learn_rate, PR_ave=TRUE) {

  b_len = length(b_true)
  n_iter = nrow(X)

  # initialize beta
  b_est = numeric(b_len)
  b_est_matrix = matrix(0, nrow = b_len, ncol = n_iter)
  est_error = numeric(n_iter)
  pred_error = numeric(n_iter)
  G = numeric(b_len)

  # loop through the most recent data point
  for (i in 1:n_iter) {

    # Update the gradient
    gradient = get_gradient(x=X[i,], y[i], b=b_est, reg=reg)
    G = G + gradient^2

    # Update beta
    b_est_raw = b_est - learn_rate * gradient /sqrt(G)
    if (PR_ave==TRUE) {
      # Polyak-Ruppert Averaging
      b_est = (b_est_raw + b_est*(i-1))/i
    } else {
      b_est = b_est_raw
    }
    # Store new beta
    b_est_matrix[, i] = b_est

    # Calculate estimation and prediction errors
    est_error[i] = est_error(b_true=b_true, b_est=b_est)
    pred_error[i] = pred_error(reg=reg, y=y, X=X, b_est=b_est)
  }

  return(list(b_est = b_est,
              # b_est_past = b_est_matrix,
              b_true = b_true,
              est_error = est_error,
              pred_error = pred_error))
}







#-----------------------------------------------------------------#
#---------  ONLINE LEARNING ALGORITHMS - HELPER FUCTIONS ---------#
#-----------------------------------------------------------------#


#1 --------- Calculate gradient of loss function ---------#
get_gradient = function(x, y, b, reg){
  # Loss function is squared error
  if (reg == "linear") {
    return (-2*x*(y - as.numeric(t(b) %*% x)))
  }
  # Loss function is log likelihood loss
  if (reg == "logistic") {
    y = 2*y-1
    return(-y*x/ ( 1+ exp(y*sum(x*b) ) ) )
  }
}

#2 --------- Calculate estimation and prediction errors ---------#
est_error = function(b_true, b_est){
  return(norm(b_true - b_est, type = "2"))
}

pred_error = function(reg, y, X, b_est) {
  if (reg=="linear") {
    # mean squared prediction error
    return(sum((y - X %*% b_est)^2)/nrow(X))
  } else if (reg=="logistic") {
    # classification error rate
    return(colMeans(sign(1/(1 + exp(-X%*%b_est))-0.5) != y))
  }
}
