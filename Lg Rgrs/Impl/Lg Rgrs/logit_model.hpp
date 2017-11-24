#pragma once

#include <functional>
#include <armadillo>

using std::function;

using arma::mat;
using arma::vec;
using arma::datum;
using arma::span;
using arma::subview;
using arma::ones;
using arma::zeros;
using arma::randn;

/* eta: learning rate factor
   lambda: regularization term factor
   eps: convergence factor

   learning_rate_t:
   - constant_rate: eta
   - error_rate: eta * norm(err, 1) / N
   - iteration_rate(init: eta): last_rate / eta

   gradient_descent_t:
   - full_descent
   - stochastic_descent

   initial_weight_t:
   - ones_weight
   - rand_weight */
struct lr_config
{
  double eta = 1.0;
  double lambda = 0.0;
  double eps = 0.0;

  enum rate
  {
    constant_rate = 0,
    error_rate,
    iteration_rate
  } learning_rate_t = constant_rate;

  enum descent
  {
    full_descent = 0,
    stochastic_descent
  } gradient_descent_t = full_descent;

  enum weight
  {
    ones_weight = 0,
    rand_weight,
    zeros_weight
  } initial_weight_t = ones_weight;
};

/* logistic regression class, mat store by rows
   why I use template functions?
   : because X can be subview<>, mat or glue<>,
   : to avoid deep copying, template function is better. */
class lr_model
{
public:
  void set_cfg(const lr_config &cfg) { this->cfg = cfg; }

  /* [X y] -> [ones X y] */
  template<class M>
  void set_data(const M &Xy) { this->oXy = join_horiz(ones(Xy.n_rows), Xy); }

  template<class M>
  static vec regression(const M &X, const vec &w) { return __logistic(join_horiz(ones(X.n_rows), X) * w); }

  template<class M>
  static vec classification(const M &X, const vec &w,
    const double &pos = 1.0, const double &neg = 0.0)
  {
    /* pX: porbility vec of X */
    vec &pX = regression(X, w);
    vec cX(pX.n_elem);
    cX.elem(find(pX > 0.5)).fill(pos);
    cX.elem(find(pX <= 0.5)).fill(neg);

    return cX;
  }

  vec train(const double &k, vec &cost)
  {
    vec w = __init[cfg.initial_weight_t](oXy.n_cols - 1);
    double eta = cfg.eta;
    cost = zeros(k);
    for (int i = 0; i < k; ++i)
    {
      /* choose subset
         oX: [ones X] in MATLAB expresion */
      auto subset = __subset[cfg.gradient_descent_t](oXy, i);
      auto oX = subset.cols(0, subset.n_cols - 2);
      auto y = subset.col(subset.n_cols - 1);

      /* gradient descent
         err: error vec
         gC: gradient(C(w, X, y), w) */
      vec err = __logistic(oX * w) - y;
      vec &gC = __gradient(err, oX, w, cfg);
      if (norm(gC) <= cfg.eps)
        break;

      /* learning rate * gradient */
      eta = __lrn_rate[cfg.learning_rate_t](err, eta, cfg);
      vec dC = eta * gC;
      if (norm(dC) <= cfg.eps)
        break;

      /* update w */
      w -= dC;

      /* new cost */
      cost(i) = __cost(y, oX, w, cfg);
      if (i > 0 && abs(cost(i) - cost(i - 1)) < cfg.eps)
        break;
    }

    return w;
  }

  /* oXy: [ones X y] in MATLAB expression */
  mat oXy;
  lr_config cfg;

private:
  /* logistic function */
  static vec __logistic(const vec &z) { return 1.0 / (1.0 + exp(-z)); }

  /* gradient function */
  template<class M>
  static vec __gradient(const vec &e, const M &X, const vec &w, const lr_config &cfg)
  {
    if (cfg.lambda == 0.0)
      return (e.t() * X).t() / e.n_elem;

    /* regt: regularization term */
    vec regt = cfg.lambda * w;
    regt(0) = 0.0;

    return ((e.t() * X).t() + regt) / e.n_elem;
  }

  /* cost function */
  template<class M>
  static double __cost(const vec &y, const M &X, const vec &w, const lr_config &cfg)
  {
    vec wX = X * w;
    auto log_prob = (y % wX) - log(1 + exp(wX));
    if (cfg.lambda == 0.0)
      return -sum(log_prob) / X.n_rows;

    /* regt: regularization term */
    vec regt = w;
    regt(0) = 0.0;

    return -(sum(log_prob) + 0.5 * cfg.lambda * dot(regt, regt)) / X.n_rows;
  }


  /* learning rate functions
     0: constant learning rate
     1: error learning rate
     2: iteration learning rate */
  function<double(const vec &, const double &, const lr_config &)> __lrn_rate[3] =
  {
    [](const vec &e, const double &last_rate, const lr_config &cfg) { return cfg.eta; },
    [](const vec &e, const double &last_rate, const lr_config &cfg) { return cfg.eta * norm(e, 1) / e.n_elem; },
    [](const vec &e, const double &last_rate, const lr_config &cfg) { return last_rate / cfg.eta; }
  };

  /* subset functions
     0: gradient descent, full mat
     1: stochastic gradient descent, iteration select vec */
  function<subview<double>(const mat &, const int &)> __subset[2] =
  {
    [](const mat &X, const int &i) { return X(span::all, span::all); },
    [](const mat &X, const int &i) { return X.row(i % X.n_rows); }
  };

  /* initialize functions
     0: ones
     1: rand */
  function<vec(const int &)> __init[3] =
  {
    [](const int &n) { return ones(n); },
    [](const int &n) { return randn(n); },
    [](const int &n) { return zeros(n); }
  };
};
