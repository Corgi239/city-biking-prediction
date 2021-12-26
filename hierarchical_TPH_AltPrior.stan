data {
  int<lower=0> N;            // Total number of observations
  int<lower=0> K;            // Number of years 
  int<lower=1,upper=K> x[N]; // Year group indicators
  vector[N] y;               // Vector of daily trip numbers
  vector[N] t;               // Vector of temperature observations
  vector[N] p;               // Vector of humidity observations
  vector[N] h;
}

parameters {
  // Hyper parameters
  real mu_alpha;
  real<lower=0> sigma_alpha;
  real mu_beta;
  real<lower=0> sigma_beta;
  real mu_gamma;
  real<lower=0> sigma_gamma;
  real mu_delta;
  real<lower=0> sigma_delta;
  
  // Model parameters
  vector[K] alpha;         // Slope along the temperature axis
  vector[K] beta;          // Slope along the precipitation axis
  vector[K] gamma;         // Slope along the humidity axis
  vector[K] delta;         // Intercept
  real<lower=0> sigma;     // Common predictive variance for all years 
}

transformed parameters {
  vector[N] mu;
  for (i in 1:N)
    mu[i] = alpha[x[i]] * t[i] + 
            beta[x[i]] * p[i] +  
            gamma[x[i]] * h[i] +
            delta[x[i]];
}

model {
  // Hyper priors
  mu_alpha ~ normal(0, 10000);
  sigma_alpha ~ cauchy(0, 500);
  mu_beta ~ normal(0, 10000);
  sigma_beta ~ cauchy(0, 500);
  mu_gamma ~ normal(0, 10000);
  sigma_gamma ~ cauchy(0, 500);
  mu_delta ~ normal(0, 10000);
  sigma_delta ~ cauchy(0, 500);
  sigma ~ cauchy(0, 500);
  
  // Priors
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);
  gamma ~ normal(mu_gamma, sigma_gamma);
  delta ~ normal(mu_delta, sigma_delta);
  
  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
}

