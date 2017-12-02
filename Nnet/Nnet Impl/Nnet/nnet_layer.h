#pragma once
#include "stdafx.h"
#include "nnet_activation.h"


namespace nnet
{
  class layer
  {
  public:
    field<activation*> acts;


    void init_malloc(const int &out_dim) { acts.set_size(out_dim); acts.fill(NULL); }


    void propagate(
      const mat &in,
      const mat &weight,
      const mat &bias,
      mat *mul,
      mat *out)
    {
      int i = 0, len = in.n_cols;

      // z = wa + b
      *mul = weight * in + repmat(bias, 1, len);

      // f(z)
      if (out->n_rows != mul->n_rows || out->n_cols != mul->n_cols)
        out->set_size(mul->n_rows, mul->n_cols);
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
      mat *in_dloss,
      mat *delta,
      mat *weight,
      mat *bias
    )
    {
      int i = 0, len = in.n_cols, in_dim = in.n_rows;

      // f'(z)
      if (delta->n_rows != mul.n_rows || delta->n_cols != mul.n_cols)
        delta->set_size(mul.n_rows, mul.n_cols);
      delta->each_row(
        [&i, this, &mul, &out_dloss](mat &r) { assert(this->acts(i) != NULL); r = this->acts(i)->back_propagate(mul.row(i)); ++i; }
      );

      // delta = loss'(f(z)) .* f'(z)
      *delta = out_dloss % *delta;

      // prev_dloss = w.t() * theta
      *in_dloss = weight->t() * *delta;

      // w = (1 - lambda) .* w - alpha .* (theta * in.t()) / len
      *weight = (1 - repmat(lambda, 1, in_dim)) % *weight - repmat(alpha, 1, in_dim) % (*delta * in.t()) / len;

      // b = b - alpha .* mean_of_rows(delta)
      *bias = *bias - alpha % mean(*delta, 1);
    }
  };
}
