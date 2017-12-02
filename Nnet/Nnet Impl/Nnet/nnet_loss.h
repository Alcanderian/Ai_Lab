#pragma once
#include "stdafx.h"


namespace nnet
{
  class loss
  {
  public:
    /* loss function */
    virtual mat avg_eval(const mat &t, const mat &y) = 0;
    /* diff of loss function */
    virtual mat diff(const mat &t, const mat &y) = 0;
  };


  class mse :
    public loss
  {
  public:
    mat avg_eval(const mat &t, const mat &y)
    {
      // mean half-square-error
      return mean(0.5 * square(y - t), 1);
    }


    mat diff(const mat &t, const mat &y)
    {
      return t - y;
    }
  };

  class xent :
    public loss
  {
  public:
    mat avg_eval(const mat &t, const mat &y)
    {
      // mean xent, -1 / n * sum_of_rows(y .* ln(t) + (1 - y) .* ln(1 - t))
      return -mean(y % log(t) + (1.0 - y) % log(1.0 - t), 1);
    }


    mat diff(const mat &t, const mat &y)
    {
      return (t - y) / (t % (1 - t));
    }
  };
}