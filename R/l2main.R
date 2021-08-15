#' Path-following algorithm for computing L2 regularized solution path
#' 
#' @description 
#' Compute \eqn{\ell_2} regularized solution path by using path-following algorithms.
#' Grid points are adaptively selected to save computation while maintaining the high accuracy of the whole path.
#' Current version only supports ridge regression (class = "square", default) and \eqn{\ell_2} regularized logistic regression (class = "logistic").
#' For both classes of problems, the method for computing solution path can be gradient descent (method = "GD), Newton method in low dimensional case (\eqn{n >= p}, method = "Newton_low")
#' and Newton method in high dimensional case (\eqn{n < p}, method = "Newton_high"). 
#' 
#' @param X A \eqn{n \times p} feature matrix.
#' @param Y A \eqn{n \times 1} response vector when class = "square", and a \eqn{n \times 1} binary vector when class = "logistic".
#' @param t_max Range of solution path \eqn{[0, t_{\max}]}.
#' @param alpha_init Initial step size specified to start the iteration.
#' @param class Specify the problem class with default value "square". Use "square" when considering ridge regression solution path, and "logistic" for \eqn{\ell_2} regularized logistic regression.
#' @param method Specify the method for computing the solution path. Default is "Newton_low". Use "GD" for gradient descent, "Newton_low" for Newton method in low dimensional case (\eqn{n >= p}) and "Newton_high" for Newton method in low dimensional case (\eqn{n < p}).
#'
#'
#' @return theta: Generated solution path
#' @return alpha_t_vec: Adaptively chosen step size
#' @return t_vec: Adaptively selected grid points
#' @return n_vec: Number of gradient steps at each selected grid point. Applicable only if method =  "GD".
#' @export
#'
l2main <- function(X, Y, t_max, alpha_init, class = "square", method = "Newton_low"){
  if(class == "square"){
    if(method == "GD"){
      .Call(`_l2path_regression_GD_path`, X, Y, t_max, alpha_init)
    } else if (method == "Newton_low"){
      .Call(`_l2path_regression_Newton_path_low_dimension_vary`, X, Y, t_max, alpha_init)
    } else if (method == "Newton_high"){
      .Call(`_l2path_regression_Newton_path_high_dimension_vary`, X, Y, t_max, alpha_init)
    } else {
      cat("\n The current version only supports 'GD', 'Newton_low' and 'Newton_high' method options.\n")
    }
  } else if (class == "logistic"){
    if(method == "GD"){
      .Call(`_l2path_logistic_GD_path`, X, Y, t_max, alpha_init)
    } else if (method == "Newton_low"){
      .Call(`_l2path_logistic_Newton_path_low_dimension_vary`, X, Y, t_max, alpha_init)
    } else if (method == "Newton_high"){
      .Call(`_l2path_logistic_Newton_path_high_dimension_vary`, X, Y, t_max, alpha_init)
    } else {
      cat("\n The current version only supports 'GD', 'Newton_low' and 'Newton_high' method options.\n")
    }
  } else {
    cat("\n The current version only supports 'square' and 'logistic' class options.\n")
  }
}
