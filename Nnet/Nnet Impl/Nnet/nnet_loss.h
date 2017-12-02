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
}