// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>


//' Solution path of ridge regression by gradient descent method
//' 
//' @description
//' Generate solution of ridge regression by using gradient descent method at each grid point.
//' Grid points are adaptively selected to save computation while maintaining the high accuracy of solution path.
//' 
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param alpha_init Initial step size specified to start the iteration
//' 
//' @details
//' This function will be called by the main function l2path, with class = "square" and method = "GD".
//' 
//' @return theta: Generated solution path
//' @return alpha_t_vec: Adaptively chosen step size
//' @return t_vec: Adaptively selected grid points
//' @return n_vec: Number of gradient steps at each grid point
// [[Rcpp::export(regression_GD_path)]]
Rcpp::List regression_GD_path(arma::mat X, arma::vec Y, double t_max, double alpha_init){
  const int n=X.n_rows;
  const int p=X.n_cols;

  // store re-usable components;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  arma::vec X_t_Y(p);
  X_t_Y=X_t*Y;

  arma::mat X_t_X(p,p);
  X_t_X=X_t*X;

  // declare the step size and time grid.
  double alpha_t=alpha_init;
  double t_GD=alpha_init;

  double epsilon=std::pow(std::exp(alpha_t)-1,2)*arma::accu(arma::square(X_t_Y/n));
  double epsilon_sqrt=std::sqrt(epsilon);

  const int N=5*std::round(t_max/alpha_init);

  // store the iteration scheme
  arma::mat theta_GD(p,N+1);
  theta_GD.zeros();

  arma::vec theta_GD_temp(p);
  theta_GD_temp.zeros();

  arma::vec theta_GD_temp_BT_LHS(p);
  theta_GD_temp_BT_LHS.zeros();

  // store the alpha vec, t vec n_t vec
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  alpha_t_vec(0)=alpha_t;

  arma::vec t_vec(N);
  t_vec.zeros();
  t_vec(0)=t_GD;

  arma::vec n_t_vec(N);
  n_t_vec.zeros();

  // declaration of variables that will be used in first layer while loop
  int index_GD=0;

  arma::vec alpha_compare_vec(3);
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=std::log(2);

  double LHS=0;
  double C1=1;
  double C2=2;

  // declaration of variables in second layer while loop: on n_t
  int index_n_t=0;
  double C0=0.1;
  double const_coef=0;

  arma::vec g_k(p);
  g_k.zeros();
  double norm_g_k=0;
  double norm_gradient_theta=0;

  // declaration of variables in third layer while loop: backtracking on eta_t
  double eta_t=1;
  double alpha_bt=0.5;
  double beta_bt=0.8;
  double fun_value_RHS=0;
  double fun_value_LHS=0;

  // begin first step;
  const_coef=C0*(std::exp(alpha_t_vec(index_GD) ) -1)/( std::exp(t_vec(index_GD) )-1);
  index_n_t=0;
  theta_GD_temp = theta_GD.col(index_GD);
  norm_gradient_theta = arma::norm(theta_GD_temp,2);
  g_k = (1-std::exp(-t_GD))/n* (X_t_X *theta_GD_temp -X_t_Y)+std::exp(-t_GD)*theta_GD_temp;
  norm_g_k = arma::norm(g_k, 2);

  while(norm_g_k>const_coef*norm_gradient_theta){
    eta_t=1;
    fun_value_RHS=(1-std::exp(-t_GD))/(2*n)*arma::accu(arma::square(Y-X*theta_GD_temp))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp % theta_GD_temp);
    theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
    fun_value_LHS=(1-std::exp(-t_GD))/(2*n)*arma::accu(arma::square(Y-X*theta_GD_temp_BT_LHS))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);

    while(fun_value_LHS>fun_value_RHS-alpha_bt*eta_t*norm_g_k*norm_g_k){
      eta_t=beta_bt*eta_t;
      theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
      fun_value_LHS=(1-std::exp(-t_GD))/(2*n)*arma::accu(arma::square(Y-X*theta_GD_temp_BT_LHS))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);
    }

    theta_GD_temp=theta_GD_temp_BT_LHS;
    norm_gradient_theta = arma::norm(theta_GD_temp,2);
    g_k = (1-std::exp(-t_GD))/n* (X_t_X *theta_GD_temp -X_t_Y)+std::exp(-t_GD)*theta_GD_temp;
    norm_g_k = arma::norm(g_k, 2);

    index_n_t=index_n_t+1;
  }

  n_t_vec(index_GD)=index_n_t;
  theta_GD.col(index_GD+1)=theta_GD_temp;
  index_GD=index_GD+1;

  LHS=C2*norm_gradient_theta*norm_gradient_theta/(std::exp(t_GD)-1);

  while(t_GD<t_max && LHS>epsilon){
    alpha_compare_vec(1)=2*alpha_t;
    alpha_compare_vec(2)=std::log(1+C1*(epsilon_sqrt*(std::exp(t_GD/2)-std::exp(-t_GD/2)) )/norm_gradient_theta );

    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_GD)=alpha_t;
    t_GD=t_GD+alpha_t;
    t_vec(index_GD)=t_GD;

    const_coef=C0*(std::exp(alpha_t_vec(index_GD) ) -1)/( std::exp(t_vec(index_GD) )-1);
    index_n_t=0;
    /*theta_GD_temp = theta_GD.col(index_GD);
    norm_gradient_theta = arma::norm(theta_GD_temp,2);*/
    g_k = (1-std::exp(-t_GD))/n* (X_t_X *theta_GD_temp -X_t_Y)+std::exp(-t_GD)*theta_GD_temp;
    norm_g_k = arma::norm(g_k, 2);

    while(norm_g_k>const_coef*norm_gradient_theta){
      eta_t=1;
      fun_value_RHS=(1-std::exp(-t_GD))/(2*n)*arma::accu(arma::square(Y-X*theta_GD_temp))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp % theta_GD_temp);
      theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
      fun_value_LHS=(1-std::exp(-t_GD))/(2*n)*arma::accu(arma::square(Y-X*theta_GD_temp_BT_LHS))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);

      while(fun_value_LHS>fun_value_RHS-alpha_bt*eta_t*norm_g_k*norm_g_k){
        eta_t=beta_bt*eta_t;
        theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
        fun_value_LHS=(1-std::exp(-t_GD))/(2*n)*arma::accu(arma::square(Y-X*theta_GD_temp_BT_LHS))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);
      }

      theta_GD_temp=theta_GD_temp_BT_LHS;
      norm_gradient_theta = arma::norm(theta_GD_temp,2);
      g_k = (1-std::exp(-t_GD))/n* (X_t_X *theta_GD_temp -X_t_Y)+std::exp(-t_GD)*theta_GD_temp;
      norm_g_k = arma::norm(g_k, 2);

      index_n_t=index_n_t+1;
    }

    n_t_vec(index_GD)=index_n_t;
    theta_GD.col(index_GD+1)=theta_GD_temp;
    index_GD=index_GD+1;

    LHS=C2*norm_gradient_theta*norm_gradient_theta/(std::exp(t_GD)-1);
  }

  arma::mat theta_GD_output(p,index_GD+1);
  theta_GD_output=theta_GD.cols(0,index_GD);

  arma::vec alpha_t_vec_output(index_GD);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_GD-1);

  arma::vec n_t_vec_output(index_GD);
  n_t_vec_output=n_t_vec.subvec(0,index_GD-1);

  arma::vec t_vec_output(index_GD+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_GD)=t_vec.subvec(0,index_GD-1);


  return Rcpp::List::create(Rcpp::Named("theta") = theta_GD_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("n_vec") = n_t_vec_output);
}

//' Solution path of ridge regression by Newton method (\eqn{n \geq p})
//' 
//' @description
//' Generate solution of ridge regression by using one-step Newton method at each grid point.
//' Grid points are adaptively selected to save computation while maintaining the high accuracy of solution path.
//' 
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param alpha_init Initial step size specified to start the iteration
//' 
//' @details
//' This function will be called by the main function l2path, with class = "square" and method = "Newton_low".
//' 
//' @return theta Generated solution path
//' @return alpha_t_vec Adaptively chosen step size
//' @return t_vec Adaptively selected grid points
// [[Rcpp::export(regression_Newton_path_low_dimension_vary)]]
Rcpp::List regression_Newton_path_low_dimension_vary(arma::mat X, arma::vec Y, double t_max, double alpha_init){
  const int n=X.n_rows;
  const int p=X.n_cols;

  // store re-usable components;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  arma::vec X_t_Y(p);
  X_t_Y=X_t*Y;

  arma::mat X_t_X(p,p);
  X_t_X=X_t*X;

  // declare the step size and time grid.
  double alpha_t=alpha_init;
  double t_Newton=alpha_init;

  double epsilon=std::pow(std::exp(alpha_t)-1,2)*arma::accu(arma::square(X_t_Y/n));
  double epsilon_sqrt=std::sqrt(epsilon);

  const int N=5*std::round(t_max/alpha_init);

  // store the iteration scheme
  arma::mat theta_Newton(p,N+1);
  theta_Newton.zeros();

  // store the alpha vec, t vec n_t vec
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  alpha_t_vec(0)=alpha_t;

  arma::vec t_vec(N);
  t_vec.zeros();
  t_vec(0)=t_Newton;

  // declaration of variables that will be used in first layer while loop
  int index_Newton=0;

  arma::vec alpha_compare_vec(4);
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=0.2;

  double LHS=0; // for path stopping criterion
  double C1=0.1;
  double C2=2;
  double beta=std::sqrt(p)*0.01;
  double norm_Newton_theta=0;

  // begin first step;
  theta_Newton.col(index_Newton+1)=arma::solve(arma::eye(p,p)+(std::exp(t_Newton)-1)/n*X_t_X, (std::exp(t_Newton)-1)/n*X_t_Y);
  index_Newton=index_Newton+1;
  norm_Newton_theta=arma::norm(theta_Newton.col(index_Newton),2);
  LHS=C2*norm_Newton_theta*norm_Newton_theta/(std::exp(t_Newton)-1);

  while(t_Newton<t_max && LHS>epsilon){
    alpha_compare_vec(1)=2*alpha_t;
    alpha_compare_vec(2)=std::log(1+(epsilon_sqrt*(std::exp(t_Newton/2)-std::exp(-t_Newton/2)) )/norm_Newton_theta );
    alpha_compare_vec(3)=std::log(1+ (std::exp(t_Newton)+std::exp(-t_Newton)-2)/(C1*beta*norm_Newton_theta));

    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_Newton=t_Newton+alpha_t;
    t_vec(index_Newton)=t_Newton;

    theta_Newton.col(index_Newton+1)=arma::solve(arma::eye(p,p)+(std::exp(t_Newton)-1)/n*X_t_X, (std::exp(t_Newton)-1)/n*X_t_Y);
    index_Newton=index_Newton+1;
    norm_Newton_theta=arma::norm(theta_Newton.col(index_Newton),2);
    LHS=C2*norm_Newton_theta*norm_Newton_theta/(std::exp(t_Newton)-1);
  }

  arma::mat theta_Newton_output(p,index_Newton+1);
  theta_Newton_output=theta_Newton.cols(0,index_Newton);

  arma::vec alpha_t_vec_output(index_Newton);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

  arma::vec t_vec_output(index_Newton+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output,
                            Rcpp::Named("t_vec") = t_vec_output);


}

//' Solution path of ridge regression by Newton method (\eqn{n < p})
//' 
//' @description
//' Generate solution of ridge regression by using one-step Newton method at each grid point. 
//' Woodbury matrix identity is applied to compute the matrix inverse.
//' Grid points are adaptively selected to save computation while maintaining the high accuracy of solution path.
//' 
//' 
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param alpha_init Initial step size specified to start the iteration
//' 
//' @details
//' This function will be called by the main function l2path, with class = "square" and method = "Newton_high".
//' 
//' @return theta Generated solution path
//' @return alpha_t_vec Adaptively chosen step size
//' @return t_vec Adaptively selected grid points
// [[Rcpp::export(regression_Newton_path_high_dimension_vary)]]
Rcpp::List regression_Newton_path_high_dimension_vary(arma::mat X, arma::vec Y, double t_max, double alpha_init){
  const int n=X.n_rows;
  const int p=X.n_cols;

  // store re-usable components;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  arma::vec X_t_Y(p);
  X_t_Y=X_t*Y;

  arma::mat X_X_t(n,n);
  X_X_t=X*X_t;

  // declare the step size and time grid.
  double alpha_t=alpha_init;
  double t_Newton=alpha_init;

  double epsilon=std::pow(std::exp(alpha_t)-1,2)*arma::accu(arma::square(X_t_Y/n));
  double epsilon_sqrt=std::sqrt(epsilon);

  const int N=5*std::round(t_max/alpha_init);

  // store the iteration scheme
  arma::mat theta_Newton(p,N+1);
  theta_Newton.zeros();

  // store the alpha vec, t vec n_t vec
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  alpha_t_vec(0)=alpha_t;

  arma::vec t_vec(N);
  t_vec.zeros();
  t_vec(0)=t_Newton;

  // declaration of variables that will be used in first layer while loop
  int index_Newton=0;

  arma::vec alpha_compare_vec(4);
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=0.2;

  double LHS=0; // for path stopping criterion
  double C1=0.1;
  double C2=2;
  double beta=std::sqrt(p)*0.01;
  double norm_Newton_theta=0;

  // begin first step;
  theta_Newton.col(index_Newton+1)=(std::exp(t_Newton)-1)/n*X_t_Y-(std::exp(t_Newton)-1)/n*X_t*arma::solve(arma::eye(n,n)+(std::exp(t_Newton)-1)/n*X_X_t, X)*(std::exp(t_Newton)-1)/n*X_t_Y;
  index_Newton=index_Newton+1;
  norm_Newton_theta=arma::norm(theta_Newton.col(index_Newton),2);
  LHS=C2*norm_Newton_theta*norm_Newton_theta/(std::exp(t_Newton)-1);

  while(t_Newton<t_max && LHS>epsilon){
    alpha_compare_vec(1)=2*alpha_t;
    alpha_compare_vec(2)=std::log(1+(epsilon_sqrt*(std::exp(t_Newton/2)-std::exp(-t_Newton/2)) )/norm_Newton_theta );
    alpha_compare_vec(3)=std::log(1+ (std::exp(t_Newton)+std::exp(-t_Newton)-2)/(C1*beta*norm_Newton_theta));

    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_Newton=t_Newton+alpha_t;
    t_vec(index_Newton)=t_Newton;

    theta_Newton.col(index_Newton+1)=(std::exp(t_Newton)-1)/n*X_t_Y-(std::exp(t_Newton)-1)/n*X_t*arma::solve(arma::eye(n,n)+(std::exp(t_Newton)-1)/n*X_X_t, X)*(std::exp(t_Newton)-1)/n*X_t_Y;
    index_Newton=index_Newton+1;
    norm_Newton_theta=arma::norm(theta_Newton.col(index_Newton),2);
    LHS=C2*norm_Newton_theta*norm_Newton_theta/(std::exp(t_Newton)-1);
  }

  arma::mat theta_Newton_output(p,index_Newton+1);
  theta_Newton_output=theta_Newton.cols(0,index_Newton);

  arma::vec alpha_t_vec_output(index_Newton);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

  arma::vec t_vec_output(index_Newton+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output,
                            Rcpp::Named("t_vec") = t_vec_output);


}

//' Solution path of \eqn{\ell_2} regularized logistic regression by gradient descent
//' 
//' @description
//' Generate solution of \eqn{\ell_2} regularized logistic regression by using gradient descent method at each grid point.
//' Grid points are adaptively selected to save computation while maintaining the high accuracy of solution path.
//' 
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} binary vector
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param alpha_init Initial step size specified to start the iteration
//' 
//' 
//' @details
//' This function will be called by the main function l2path, with class = "logistic" and method = "GD".
//' 
//' @return theta Generated solution path
//' @return alpha_t_vec Adaptively chosen step size
//' @return t_vec Adaptively selected grid points
//' @return n_vec Number of gradient steps at each grid point
// [[Rcpp::export(logistic_GD_path)]]
Rcpp::List logistic_GD_path(arma::mat X, arma::vec Y, double t_max, double alpha_init) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  double alpha_t=alpha_init;
  double t_step=alpha_init;

  double epsilon=0;
  epsilon=std::pow(std::exp(alpha_t)-1,2)*arma::accu(arma::square(X_t*Y/(2*n)));

  const int N=5*std::round(t_max/alpha_init);

  // store the iteration scheme
  arma::mat theta_GD(p,N+1);
  theta_GD.zeros();

  arma::vec theta_GD_temp(p);
  theta_GD_temp.zeros();

  arma::vec theta_GD_temp_BT_LHS(p);
  theta_GD_temp_BT_LHS.zeros();

  arma::vec b_GD(n);
  b_GD.zeros();

  // store the alpha vec, t vec n_t vec
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  alpha_t_vec(0)=alpha_t;

  arma::vec t_vec(N);
  t_vec.zeros();
  t_vec(0)=t_step;

  arma::vec n_t_vec(N);
  n_t_vec.zeros();

  // declaration of variables that will be used in first layer while loop
  int index_n_t=0;
  int index_GD=0;
  double t_GD=0;

  arma::vec alpha_compare_vec(3);
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=std::log(2);

  double LHS=0;

  // declaration of variables in second layer while loop: on n_t
  double C0=12;

  arma::vec g_k(p);
  g_k.zeros();
  double norm_g_k=0;
  double norm_gradient_theta=0;

  // declaration of variables in third layer while loop: backtracking on eta_t
  double eta_t=1;
  double alpha_bt=0.5;
  double beta_bt=0.8;
  double fun_value_RHS=0;
  double fun_value_LHS=0;

  // begin first step;
  index_n_t=0;
  t_GD=t_step;
  theta_GD_temp = theta_GD.col(index_GD);
  norm_gradient_theta = arma::norm(theta_GD_temp,2);
  b_GD = 1/(1+arma::exp(Y%(X*theta_GD_temp)) );
  g_k = std::exp(-t_GD)*theta_GD_temp-(1-std::exp(-t_GD))/n*X_t*(b_GD%Y);
  norm_g_k = arma::norm(g_k, 2);

  while(norm_g_k>(std::exp(alpha_t_vec(index_GD) ) -1)/(C0*( std::exp(t_vec(index_GD))-1) )*norm_gradient_theta){
     eta_t=1;
     fun_value_RHS=(1-std::exp(-t_GD))/n*arma::accu(arma::log(1+arma::exp(-Y%(X*theta_GD_temp)) ))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp % theta_GD_temp);
     theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
     fun_value_LHS=(1-std::exp(-t_GD))/n*arma::accu(arma::log(1+arma::exp(-Y%(X*theta_GD_temp_BT_LHS)) ))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);

     while(fun_value_LHS>fun_value_RHS-alpha_bt*eta_t*norm_g_k*norm_g_k){
       eta_t=beta_bt*eta_t;
       theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
       fun_value_LHS=(1-std::exp(-t_GD))/n*arma::accu(arma::log(1+arma::exp(-Y%(X*theta_GD_temp_BT_LHS)) ))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);
     }

     theta_GD_temp=(1-eta_t*std::exp(-t_GD))*theta_GD_temp+eta_t*(1-std::exp(-t_GD))/n*X_t*(b_GD%Y);
     norm_gradient_theta = arma::norm(theta_GD_temp,2);
     b_GD = 1/(1+arma::exp(Y%(X*theta_GD_temp)) );
     g_k = std::exp(-t_GD)*theta_GD_temp-(1-std::exp(-t_GD))/n*X_t*(b_GD%Y);
     norm_g_k = arma::norm(g_k, 2);

     index_n_t=index_n_t+1;
  }

  n_t_vec(index_GD)=index_n_t;
  theta_GD.col(index_GD+1)=theta_GD_temp;
  index_GD=index_GD+1;

  LHS=2*arma::accu(arma::square(theta_GD.col(index_GD)))/(std::exp(t_step)-1);

  while(t_step<t_max && LHS>epsilon){
      norm_gradient_theta=arma::norm(theta_GD.col(index_GD),2);
      alpha_compare_vec(1)=2*alpha_t;
      alpha_compare_vec(2)=std::log(1+(std::sqrt(epsilon)*(std::exp(t_step/2)-std::exp(-t_step/2)) )/norm_gradient_theta );

      alpha_t=alpha_compare_vec.min();
      alpha_t_vec(index_GD)=alpha_t;
      t_step=t_step+alpha_t;
      t_vec(index_GD)=t_step;

      index_n_t=0;
      t_GD=t_step;
      theta_GD_temp = theta_GD.col(index_GD);
      norm_gradient_theta = arma::norm(theta_GD_temp,2);
      b_GD = 1/(1+arma::exp(Y%(X*theta_GD_temp)) );
      g_k = std::exp(-t_GD)*theta_GD_temp-(1-std::exp(-t_GD))/n*X_t*(b_GD%Y);
      norm_g_k = arma::norm(g_k, 2);

      while(norm_g_k>(std::exp(alpha_t_vec(index_GD) ) -1)/(C0*( std::exp(t_vec(index_GD))-1) )*norm_gradient_theta){
          eta_t=1;
          fun_value_RHS=(1-std::exp(-t_GD))/n*arma::accu(arma::log(1+arma::exp(-Y%(X*theta_GD_temp)) ))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp % theta_GD_temp);
          theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
          fun_value_LHS=(1-std::exp(-t_GD))/n*arma::accu(arma::log(1+arma::exp(-Y%(X*theta_GD_temp_BT_LHS)) ))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);

          while(fun_value_LHS>fun_value_RHS-alpha_bt*eta_t*norm_g_k*norm_g_k){
              eta_t=beta_bt*eta_t;
              theta_GD_temp_BT_LHS=theta_GD_temp-eta_t*g_k;
              fun_value_LHS=(1-std::exp(-t_GD))/n*arma::accu(arma::log(1+arma::exp(-Y%(X*theta_GD_temp_BT_LHS)) ))+std::exp(-t_GD)/2*arma::accu(theta_GD_temp_BT_LHS % theta_GD_temp_BT_LHS);
          }

          theta_GD_temp=(1-eta_t*std::exp(-t_GD))*theta_GD_temp+eta_t*(1-std::exp(-t_GD))/n*X_t*(b_GD%Y);
          norm_gradient_theta = arma::norm(theta_GD_temp,2);
          b_GD = 1/(1+arma::exp(Y%(X*theta_GD_temp)) );
          g_k = std::exp(-t_GD)*theta_GD_temp-(1-std::exp(-t_GD))/n*X_t*(b_GD%Y);
          norm_g_k = arma::norm(g_k, 2);

          index_n_t=index_n_t+1;
      }

      n_t_vec(index_GD)=index_n_t;
      theta_GD.col(index_GD+1)=theta_GD_temp;
      index_GD=index_GD+1;

      LHS=2*arma::accu(arma::square(theta_GD.col(index_GD)))/(std::exp(t_step)-1);
  }

  arma::mat theta_GD_output(p,index_GD+1);
  theta_GD_output=theta_GD.cols(0,index_GD);

  arma::vec alpha_t_vec_output(index_GD);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_GD-1);

  arma::vec n_t_vec_output(index_GD);
  n_t_vec_output=n_t_vec.subvec(0,index_GD-1);

  arma::vec t_vec_output(index_GD+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_GD)=t_vec.subvec(0,index_GD-1);


  return Rcpp::List::create(Rcpp::Named("theta") = theta_GD_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("n_vec") = n_t_vec_output);

}

//' Solution path of \eqn{\ell_2} regularized logistic regression by Newton method (\eqn{n \geq p})
//' 
//' @description
//' Generate solution of \eqn{\ell_2} regularized logistic regression by using one-step Newton method at each grid point. 
//' Grid points are adaptively selected to save computation while maintaining the high accuracy of solution path.
//' 
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} binary vector
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param alpha_init Initial step size specified to start the iteration
//' 
//' @details
//' This function will be called by the main function l2path, with class = "logistic" and method = "Newton_low".
//' 
//' @return theta Generated solution path
//' @return alpha_t_vec Adaptively chosen step size
//' @return t_vec Adaptively selected grid points
// [[Rcpp::export(logistic_Newton_path_low_dimension_vary)]]
Rcpp::List logistic_Newton_path_low_dimension_vary(arma::mat X, arma::vec Y, double t_max, double alpha_init){
   const int n=X.n_rows;
   const int p=X.n_cols;
   arma::mat X_t(p,n);
   X_t=arma::trans(X);
   double norm_gradient_L0=0;
   norm_gradient_L0=arma::norm(X_t*Y,2)/(2*n);

   const int N=5*std::round(t_max/alpha_init);
   const double C1=0.01; // original 0.01
   double beta=std::sqrt(p)*0.01;

   double norm_gradient_theta=0;
   arma::vec alpha_compare_vec(4);
   alpha_compare_vec.zeros();
   alpha_compare_vec(0)=2*alpha_init;

   double t_Newton=0;
   arma::mat temp_Newton_1(p,n);
   temp_Newton_1.zeros();
   arma::mat temp_Newton_2(n,p);
   temp_Newton_2.zeros();
   arma::mat temp_Newton_3(p,p);
   temp_Newton_3.zeros();
   arma::vec temp_Newton_4(p);
   temp_Newton_4.zeros();

   arma::vec b_Newton(n);
   b_Newton.zeros();

   double alpha_t=alpha_init;
   double t_step=alpha_init;

   arma::mat theta_Newton(p,N+1);
   theta_Newton.zeros();

   arma::vec alpha_t_vec(N);
   alpha_t_vec.zeros();
   alpha_t_vec(0)=alpha_t;

   arma::vec t_vec(N);
   t_vec.zeros();
   t_vec(0)=t_step;

   int index_Newton=0;
   t_Newton=t_step;

   b_Newton=1/(1+arma::exp( Y%(X*theta_Newton.col(index_Newton) ) ) );
   temp_Newton_1=X_t.each_row() % arma::trans(b_Newton);
   temp_Newton_2=(1-b_Newton)%X.each_col();
   temp_Newton_3=(std::exp(t_Newton)-1)/n*temp_Newton_1*temp_Newton_2+arma::eye(p,p);
   temp_Newton_4=(std::exp(t_Newton)-1)/n*X_t*(b_Newton%Y)-theta_Newton.col(index_Newton);
   theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)+arma::solve(temp_Newton_3,temp_Newton_4);

   index_Newton=index_Newton+1;

   while(t_step<t_max){
    // update alpha;
    norm_gradient_theta=arma::norm(theta_Newton.col(index_Newton),2);
    alpha_compare_vec(1)=2*alpha_t;
    alpha_compare_vec(2)=std::log(1+ (std::exp(alpha_init)-1)*norm_gradient_L0*(std::exp(t_step/2)-std::exp(-t_step/2))/norm_gradient_theta );
    alpha_compare_vec(3)=std::log(1+ (1-std::exp(-t_step))/(C1*beta*norm_gradient_theta));

    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_Newton)=t_step;

    t_Newton=t_step;
    b_Newton=1/(1+arma::exp( Y%(X*theta_Newton.col(index_Newton) ) ) );
    temp_Newton_1=X_t.each_row() % arma::trans(b_Newton);
    temp_Newton_2=(1-b_Newton)%X.each_col();
    temp_Newton_3=(std::exp(t_Newton)-1)/n*temp_Newton_1*temp_Newton_2+arma::eye(p,p);
    temp_Newton_4=(std::exp(t_Newton)-1)/n*X_t*(b_Newton%Y)-theta_Newton.col(index_Newton);
    theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)+arma::solve(temp_Newton_3,temp_Newton_4);

    index_Newton=index_Newton+1;
   }

   arma::mat theta_Newton_output(p,index_Newton+1);
   theta_Newton_output=theta_Newton.cols(0,index_Newton);

   arma::vec alpha_t_vec_output(index_Newton);
   alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

   arma::vec t_vec_output(index_Newton+1);
   t_vec_output(0)=0;
   t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output,
                            Rcpp::Named("t_vec") = t_vec_output);

}

//' Solution path of \eqn{\ell_2} regularized logistic regression by Newton method (\eqn{n < p})
//' 
//' @description
//' Generate solution of \eqn{\ell_2} regularized logistic regression by using one-step Newton method at each grid point. 
//' Woodbury matrix identity is applied to compute the matrix inverse.
//' Grid points are adaptively selected to save computation while maintaining the high accuracy of solution path.
//' 
//' 
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} binary vector
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param alpha_init Initial step size specified to start the iteration
//' 
//' @details
//' This function will be called by the main function l2path, with class = "logistic" and method = "Newton_high".
//' 
//' @return theta Generated solution path
//' @return alpha_t_vec Adaptively chosen step size
//' @return t_vec Adaptively selected grid points
// [[Rcpp::export(logistic_Newton_path_high_dimension_vary)]]
Rcpp::List logistic_Newton_path_high_dimension_vary(arma::mat X, arma::vec Y, double t_max, double alpha_init){
   const int n=X.n_rows;
   const int p=X.n_cols;
   arma::mat X_t(p,n);
   X_t=arma::trans(X);
   double norm_gradient_L0=0;
   norm_gradient_L0=arma::norm(X_t*Y,2)/(2*n);

   const int N=5*std::round(t_max/alpha_init);
   const double C1=0.01;
   double beta=std::sqrt(p)*0.01;

   double norm_gradient_theta=0;
   arma::vec alpha_compare_vec(4);
   alpha_compare_vec.zeros();
   alpha_compare_vec(0)=2*alpha_init;

   double t_Newton=0;
   arma::mat temp_Newton_1(p,n);
   temp_Newton_1.zeros();
   arma::mat temp_Newton_2(n,p);
   temp_Newton_2.zeros();
   arma::mat temp_Newton_3(p,p);
   temp_Newton_3.zeros();
   arma::vec temp_Newton_4(p);
   temp_Newton_4.zeros();

   arma::vec b_Newton(n);
   b_Newton.zeros();

   double alpha_t=alpha_init;
   double t_step=alpha_init;

   arma::mat theta_Newton(p,N+1);
   theta_Newton.zeros();

   arma::vec alpha_t_vec(N);
   alpha_t_vec.zeros();
   alpha_t_vec(0)=alpha_t;

   arma::vec t_vec(N);
   t_vec.zeros();
   t_vec(0)=t_step;

   int index_Newton=0;
   t_Newton=t_step;

   b_Newton=1/(1+arma::exp(Y%(X*theta_Newton.col(index_Newton) ) ) );
   temp_Newton_1=X_t.each_row() % arma::trans(b_Newton);
   temp_Newton_2=(1-b_Newton)%X.each_col();
   temp_Newton_3=(std::exp(t_Newton)-1)/n*temp_Newton_1*arma::solve(arma::eye(n,n)+(std::exp(t_Newton)-1)/n*temp_Newton_2*temp_Newton_1, temp_Newton_2);
   temp_Newton_4=(std::exp(t_Newton)-1)/n*X_t*(b_Newton%Y)-theta_Newton.col(index_Newton);
   theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)+ temp_Newton_4-temp_Newton_3*temp_Newton_4;

   index_Newton=index_Newton+1;

   while(t_step<t_max){
    // update alpha;
    norm_gradient_theta=arma::norm(theta_Newton.col(index_Newton),2);
    alpha_compare_vec(1)=2*alpha_t;
    alpha_compare_vec(2)=std::log(1+ (std::exp(alpha_init)-1)*norm_gradient_L0*(std::exp(t_step/2)-std::exp(-t_step/2))/norm_gradient_theta );
    alpha_compare_vec(3)=std::log(1+ (1-std::exp(-t_step))/(C1*beta*norm_gradient_theta));

    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_Newton)=t_step;

    t_Newton=t_step;
    b_Newton=1/(1+arma::exp(Y%(X*theta_Newton.col(index_Newton) ) ) );
    temp_Newton_1=X_t.each_row() % arma::trans(b_Newton);
    temp_Newton_2=(1-b_Newton)%X.each_col();
    temp_Newton_3=(std::exp(t_Newton)-1)/n*temp_Newton_1*arma::solve(arma::eye(n,n)+(std::exp(t_Newton)-1)/n*temp_Newton_2*temp_Newton_1, temp_Newton_2);
    temp_Newton_4=(std::exp(t_Newton)-1)/n*X_t*(b_Newton%Y)-theta_Newton.col(index_Newton);
    theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)+ temp_Newton_4-temp_Newton_3*temp_Newton_4;

    index_Newton=index_Newton+1;
   }

   arma::mat theta_Newton_output(p,index_Newton+1);
   theta_Newton_output=theta_Newton.cols(0,index_Newton);

   arma::vec alpha_t_vec_output(index_Newton);
   alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

   arma::vec t_vec_output(index_Newton+1);
   t_vec_output(0)=0;
   t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output,
                            Rcpp::Named("t_vec") = t_vec_output);

}
