#pragma once
#include "stdafx.h"


namespace nnet
{
  class loss
  {
  public:
    /* loss function */
    virtual void avg_eval(const mat &t, const mat &y, mat *e) = 0;
    /* diff of loss function */
    virtual void diff(const mat &t, const mat &y, mat *d) = 0;
  };


  class mse :
    public loss
  {
  public:
    // mean half-square-error
    void avg_eval(const mat &t, const mat &y, mat *e) { *e = mean(0.5 * square(y - t), 1); }
    void diff(const mat &t, const mat &y, mat *d) { *d = t - y; }
  };

  class xent :
    public loss
  {
  public:
    // xent: -1 / n * sum_of_rows(y .* ln(t) + (1 - y) .* ln(1 - t))
    void avg_eval(const mat &t, const mat &y, mat *e) { *e = -mean(y % log(t) + (1.0 - y) % log(1.0 - t), 1); }
    void diff(const mat &t, const mat &y, mat *d) { *d = (t - y) / (t % (1.0 - t)); }
  };

  class nf1 :
    public loss
  {
  public:
    // tag actual predict state(actual + 2 * predict)
    // tp     1        1         3
    // fn     1        0         1
    // tn     0        0         0
    // fp     0        1         2
    void avg_eval(const mat &t, const mat &y, mat *e)
    {
      if (e->n_rows != y.n_rows)
        e->set_size(y.n_rows, 1);

      mat h = t;
      h.elem(find(h >= 0.5)).fill(1.0);
      h.elem(find(h < 0.5)).fill(0.0);

      for (int i = 0; i < y.n_rows; ++i)
      {
        mat state = y.row(i) + 2 * h.row(i);
        double tp = uvec(find(state == 3.0)).n_elem;
        double fn = uvec(find(state == 1.0)).n_elem;
        double tn = uvec(find(state == 0.0)).n_elem;
        double fp = uvec(find(state == 2.0)).n_elem;

        double recall = tp / (tp + fn);
        double precision = tp / (tp + fp);
        double f1 = 2 * precision * recall / (precision + recall);
        e->at(i, 0) = -f1;
      }
    }

    void diff(const mat &t, const mat &y, mat *d)
    {
      assert(false);
    }
  };
}
