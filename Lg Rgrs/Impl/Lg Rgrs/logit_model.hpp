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
using arma::randn;

/* alpha: learning rate factor
   lambda: regularization term factor
   eps: convergence factor
   
   learning_rate_t:
   - constant_rate: alpha
   - error_rate: alpha * norm(err, 1) / N
   - iteration_rate(init: alpha): last_rate / alpha
   
   gradient_descent_t:
   - full_descent
   - stochastic_descent

   initial_weight_t:
   - ones_weight
   - rand_weight */
struct lr_config
{
  double alpha = 1.0;
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
    rand_weight
  } initial_weight_t = ones_weight;
};

/* logistic regression class,
   mat store by rows */
class lr_model
{
public:
  void set_cfg(const lr_config &cfg) { this->cfg = cfg; }
  
  template<class M>
  static vec regression(const M &X, const vec &w)
  {
    return __logistic(join_horiz(ones(X.n_rows), X) * w);
  }

  template<class M>
  static vec classification(const M &X, const vec &w,
    const double &pos = 1.0, const double &neg = 0.0)
  {
    /* pX: porbility vec of X */
    vec pX = regression(X, w);
    vec cX(pX.n_elem);
    cX.elem(find(pX > 0.5)).fill(pos);
    cX.elem(find(pX <= 0.5)).fill(neg);

    return cX;
  }

  /* [X y] -> [ones X y] */
  template<class M>
  void set_data(const M &Xy)
  {
    this->oXy = join_horiz(ones(Xy.n_rows), Xy);
  }

  vec train(const double &k)
  {
    vec last_w, w = __init[cfg.initial_weight_t](oXy.n_cols - 1);
    double rate = cfg.alpha;
    for (int i = 0; i < k; ++i)
    {
      /* choose subset
         oX: [ones X] in MATLAB expresion */
      auto subset = __subset[cfg.gradient_descent_t](oXy, i);
      auto oX = subset.cols(0, oXy.n_cols - 2);
      auto y = subset.col(oXy.n_cols - 1);
      
      /* gradient descent
         err: error vec
         gX: gradient(C(w, X, y), w) */
      vec err = __logistic(oX * w) - y;
      vec &gX = __gradient(err, oX, w, cfg);
      if (norm(gX) < cfg.eps)
        break;

      /* update w */
      last_w = w;
      rate = __lrn_rate[cfg.learning_rate_t](err, rate, cfg);
      w -= rate * gX;
      if (norm(w - last_w) < cfg.eps)
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

  /* gradient function
     why I use template function?
     : because X can be subview<>, mat or glue<>,
     : to avoid deep copying, template function is better. */
  template<class M>
  static vec __gradient(const vec &e, const M &X,
    const vec &w, const lr_config &cfg)
  {
    /* regt: regularization term */
    double regt = (cfg.lambda == 0.0) ? 0.0 : cfg.lambda * sum(w);
    
    return ((e.t() * X).t() + regt) / e.n_elem;
  }

  /* learning rate functions
     0: constant learning rate
     1: error learning rate
     2: iteration learning rate */
  function<double(const vec &, const double &, const lr_config &)>
    __lrn_rate[3] =
  {
    [](const vec &e, const double &last_rate, const lr_config &cfg)
    {
      return cfg.alpha;
    },
    [](const vec &e, const double &last_rate, const lr_config &cfg)
    {
      return cfg.alpha * norm(e, 1) / e.n_elem;
    },
    [](const vec &e, const double &last_rate, const lr_config &cfg)
    {
      return last_rate / cfg.alpha;
    }
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
  function<vec(const int &)> __init[2] =
  {
    [](const int &n) { return ones(n); },
    [](const int &n) { return randn(n); }
  };
};
