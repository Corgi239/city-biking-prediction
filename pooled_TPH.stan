data {
  int<lower=0> N;      // Number of observations
  real y[N];           // Vector of daily trip numbers
  real t[N];           // Vector of temperature observations
  real p[N];           // Vector of precipitation observations
  real h[N];           // Vector of humidity observations
}

parameters {
  real alpha;          // Slope parameter along the temperature axis
  real beta;           // Slope parameter along the precipitatin axis
  real gamma;          // Slope parameter along the humidity axis
  real delta;          // Intercept
  real<lower=0> sigma; // Variance
}

transformed parameters {
  real mu[N];
  for (i in 1:N) mu[i] = alpha * t[i] + beta * p[i] + gamma * h[i] + delta;
}

model {
  // Priors
  alpha ~ normal(1840, 1118.54);
  beta ~ normal(-1400, 851.06);
  gamma ~ normal(-1750,1063.83);
  delta ~ normal(0, 10000);
  sigma ~ cauchy(0, 500);
  
  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
