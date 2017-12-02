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
      int len = y.n_cols;

      return 0.5 * pow(y - t, 2) * ones(1, len).t() / len;
    }


    mat diff(const mat &t, const mat &y)
    {
      return t - y;
    }
  };
}