#pragma once
#include "stdafx.h"
#include "nnet_activation.h"
#include "nnet_optimizer.h"


namespace nnet
{
  class layer
  {
  public:
    field<activation*> acts;
    optimizer* weight_opt = NULL;
    optimizer* bias_opt = NULL;

    void init_malloc(const int &out_dim) { acts.set_size(out_dim); acts.fill(NULL); }


    void propagate(
      const mat &in,
      const mat &weight,
      const mat &bias,
      mat *mul,
      mat *out)
    {
      int len = in.n_cols;

      // z = wa + b
      *mul = weight * in + repmat(bias, 1, len);

      // f(z)
      if (out->n_rows != mul->n_rows || out->n_cols != mul->n_cols)
        out->set_size(mul->n_rows, mul->n_cols);
      int i = 0;
      out->each_row(
        [&i, this, mul](mat &r) { assert(this->acts(i) != NULL); r = this->acts(i)->propagate(mul->row(i)); ++i; }
      );
    }


    void back_propagate(
      const mat &in,
      const mat &mul,
      const mat &out_dloss,
      const mat &alpha,
      const mat &lambda,
      const double &k,
      mat *in_dloss,
      mat *delta,
      mat *weight,
      mat *bias
    )
    {
      assert(weight_opt != NULL);
      assert(bias_opt != NULL);

      int len = in.n_cols, in_dim = in.n_rows;

      // f'(z)
      if (delta->n_rows != mul.n_rows || delta->n_cols != mul.n_cols)
        delta->set_size(mul.n_rows, mul.n_cols);
      int i = 0;
      delta->each_row(
        [&i, this, &mul, &out_dloss](mat &r) { assert(this->acts(i) != NULL); r = this->acts(i)->back_propagate(mul.row(i)); ++i; }
      );

      // delta = loss'(f(z)) .* f'(z)
      *delta = out_dloss % *delta;

      // prev_dloss = w.t() * theta
      *in_dloss = weight->t() * *delta;

      // w =  w - alpha .* (opt((theta * in.t()) / len) + lambda .* w)
      *weight = *weight - repmat(alpha, 1, in_dim) % weight_opt->optimize((*delta * in.t()) / len + repmat(lambda, 1, in_dim) % *weight, k);

      // b = b - alpha .* opt(mean_of_rows(delta))
      *bias = *bias - alpha % bias_opt->optimize(mean(*delta, 1), k);
    }
  };
}
