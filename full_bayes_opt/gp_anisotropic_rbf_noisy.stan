functions {
  real ARD_kernel(vector u,
                  vector v,
                  real sq_alpha,
                  vector rho){
    return sq_alpha * exp(-0.5 * dot_self((u - v) ./ rho));
  }
  matrix cov_exp_quad_ARD_self(vector[] x,
                               vector y_var,
                               real alpha,
                               vector rho,
                               real delta){
    int N = size(x);
    matrix[N, N] Sigma;

    real sq_alpha = square(alpha);
    for (i in 1:(N - 1)) {
      Sigma[i, i] = y_var[i] + sq_alpha + delta;
      for (j in (i + 1):N) {
        Sigma[i, j] = ARD_kernel(x[i], x[j], sq_alpha, rho);
        Sigma[j, i] = Sigma[i, j];
      }
    }
    Sigma[N, N] = sq_alpha + delta;
    return Sigma;
  }
  matrix cov_exp_quad_ARD_other(vector[] x,
                                vector[] x_tilde,
                                real alpha,
                                vector rho){
    int N = size(x);
    int M = size(x_tilde);
    matrix[N, M] K;

    real sq_alpha = square(alpha);
    for (i in 1:N) {
      for (j in 1:M) {
        K[i, j] = ARD_kernel(x[i], x_tilde[j], sq_alpha, rho);
      }
    }
    return K;
  }
  vector gp_pred_rng(matrix L_Sigma,
                     vector[] x_tilde,
                     vector y,
                     vector[] x,
                     real alpha,
                     vector rho,
                     real delta) {
    int N = rows(y);
    int M = size(x_tilde);
    vector[M] g;

    {
      matrix[N, M] K;            // Covariance of x with x_tilde
      matrix[N, M] X;            // helper variable: X = L^{-1}K; storing it obviates an additional mdivide_* operation.
      vector[N] z;               // L^{-1} y
      vector[M] g_mu;            // mean vector for predictions y_tilde
      matrix[M, M] g_cov;        // covariance matrix for y_tilde
      matrix[M, M] diag_delta;   // small amount of noise on the diagonal helps fitting
      vector[M] y_tilde_var = rep_vector(0.0, M);     // known measurement error of y_tilde, i.e. 0.0.

      K = cov_exp_quad_ARD_other(x, x_tilde, alpha, rho);

      z = mdivide_left_tri_low(L_Sigma, y);
      X = mdivide_left_tri_low(L_Sigma, K);
      g_mu = X' * z;
      g_cov = cov_exp_quad_ARD_self(x_tilde, y_tilde_var, alpha, rho, delta) - X' * X;

      g = multi_normal_rng(g_mu, g_cov);
    }
    return g;
  }
}
data {
  int<lower=1> N;
  int<lower=1> M;
  int<lower=1> D;
  vector[D] x[N];
  vector[D] x_tilde[M];
  vector[N] y;
  vector<lower=0.0>[N] y_var;
}
transformed data {
  real delta = 1e-9;
  real y_bar = mean(y);
  vector[N] y_shift = y - y_bar;
}
parameters {
  vector<lower=0>[D] rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  vector[N] eta;
}
transformed parameters{
  vector[N] f;
  matrix[N, N] L_Sigma;

  {
    matrix[N, N] Sigma = cov_exp_quad_ARD_self(x, y_var, alpha, rho, delta);

    L_Sigma = cholesky_decompose(Sigma);
    f = L_Sigma * eta;
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 0.1);
  eta ~ normal(0, 1);
  y_shift ~ normal(f, sigma);
}
generated quantities {
  vector[M] g;
  vector[M] y_tilde;

  g = gp_pred_rng(L_Sigma, x_tilde, y_shift, x, alpha, rho, delta);
  for (m in 1:M)
    y_tilde[m] = y_bar + normal_rng(g[m], sigma);
}
