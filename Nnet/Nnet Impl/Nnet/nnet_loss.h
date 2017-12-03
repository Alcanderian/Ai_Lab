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
    void avg_eval(const mat &t, const mat &y, mat *e)
    {
      // mean half-square-error
      *e = mean(0.5 * square(y - t), 1);
    }


    void diff(const mat &t, const mat &y, mat *d)
    {
      *d = t - y;
    }
  };

  class xent :
    public loss
  {
  public:
    void avg_eval(const mat &t, const mat &y, mat *e)
    {
      // mean xent, -1 / n * sum_of_rows(y .* ln(t) + (1 - y) .* ln(1 - t))
      *e = -mean(y % log(t) + (1.0 - y) % log(1.0 - t), 1);
    }


    void diff(const mat &t, const mat &y, mat *d)
    {
      *d = (t - y) / (t % (1 - t));
    }
  };
}