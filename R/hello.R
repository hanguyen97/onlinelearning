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
