#pragma once
#include "stdafx.h"
#include "nnet_activation.h"
#include "nnet_optimizer.h"


namespace nnet
{
  class layer
  {
  public:
    activation* act = NULL;
    optimizer* weight_opt = NULL;
    optimizer* bias_opt = NULL;


    void propagate(
      const mat &in,
      const mat &weight,
      const mat &bias,
      mat *out)
    {
      assert(act != NULL);

      int len = in.n_cols;

      // f(z = wa + b)
      act->propagate(weight * in + repmat(bias, 1, len), out);
    }


    void back_propagate(
      const mat &in,
      const mat &out,
      const mat &out_dloss,
      const mat &alpha,
      const mat &lambda,
      const double &k,
      mat *in_dloss,
      mat *delta,
      mat *weight_gradient,
      mat *bias_gradient,
      mat *weight,
      mat *bias
    )
    {
      assert(act != NULL);
      assert(weight_opt != NULL);
      assert(bias_opt != NULL);

      int len = in.n_cols, in_dim = in.n_rows;

      // f'(z)
      act->back_propagate(out, delta);

      // delta = loss'(f(z)) .* f'(z)
      *delta = out_dloss % *delta;

      // prev_dloss = w.t() * delta
      *in_dloss = weight->t() * *delta;

      // w =  w - alpha .* opt((delta * in.t()) / len + lambda .* w)
      *weight_gradient = (*delta * in.t()) / len + repmat(lambda, 1, in_dim) % *weight;
      weight_opt->optimize(k, weight_gradient);
      *weight = *weight - repmat(alpha, 1, in_dim) % *weight_gradient;

      // b = b - alpha .* opt(mean_of_rows(delta))
      *bias_gradient = mean(*delta, 1);
      bias_opt->optimize(k, bias_gradient);
      *bias = *bias - alpha % *bias_gradient;
    }
  };
}
