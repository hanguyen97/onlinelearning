#-----------------------------------------------------------------#
#---------  ONLINE LEARNING ALGORITHMS  --------------------------#
#-----------------------------------------------------------------#

#' Sample i.i.d data
#'
#' @param reg sample data from either [reg="linear"] or [reg = "logistic"]
#' @param n sample size
#' @param b_len length of coefficient beta vector
#' @param bn0_len length of non-zero betas
#' @param signal standard deviation of an independent Gaussian distribution with mean 0
#' from which the non-zero betas are drawn from
#' @param noise standard deviation of a normal distribution with mean 0
#' from which the error vector is drawn from
#'
#' @returns test return
#' @export
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

#' Title 1
#'
#' @param reg test
#' @param y test
#' @param X test
#' @param b_true test
#' @param learn_rate test
#' @param PR_ave test
#'
#' @return test
#' @export
run_OGD <- function(reg, y, X, b_true, learn_rate, PR_ave=TRUE) {

  b_len = length(b_true)
  n_iter = nrow(X)

  # initialize beta
  b_est = runif(b_len, 0, 1)
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
              b_est_past = b_est_matrix,
              b_true = b_true,
              est_error = est_error,
              pred_error = pred_error))
}


#' Title 3
#'
#' @param reg test
#' @param y test
#' @param X test
#' @param b_true test
#' @param learn_rate test
#' @param PR_ave test
#'
#' @return test
#' @export
run_AdaGrad <- function(reg, y, X, b_true, learn_rate, PR_ave=TRUE) {

  b_len = length(b_true)
  n_iter = nrow(X)

  # initialize beta
  b_est = runif(b_len, 0, 1)
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
              b_est_past = b_est_matrix,
              b_true = b_true,
              est_error = est_error,
              pred_error = pred_error))
}
