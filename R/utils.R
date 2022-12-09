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

#2 --------- Calculate gradient of mirror map function ---------#
# phi_bt = 1/2 (p-norm of b_t)^2
# https://math.stackexchange.com/questions/1482494/derivative-of-the-l-p-norm
gradient_mp <- function(b) {
  p = 1 + 1/log(length(b))
  pnorm_b = (sum(abs(b)^p))^(1/p)
  return( (abs(b)/pnorm_b)^(p-1) * sign(b) )
}

#3 --------- Calculate inverse of gradient of mirror map function ---------#
invgd_mp <- function(b) {
  q = 1 + log(length(b))
  denominator = ((sum(abs(b)^q))^(1/q))^(q-2)
  return( (abs(b)^(q-1))*sign(b)/denominator )
}

#4 --------- Calculate estimation and prediction errors ---------#
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
