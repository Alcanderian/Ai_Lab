#pragma once
#include "stdafx.h"


namespace nnet
{
  class loss
  {
  public:
    /* loss function */
    virtual mat eval(const mat &t, const mat &y) = 0;
    /* diff of loss function */
    virtual mat diff(const mat &t, const mat &y) = 0;
  };


  class mse :
    public loss
  {
  public:
    mat eval(const mat &t, const mat &y)
    {
      return 0.5 * pow(y - t, 2);
    }


    mat diff(const mat &t, const mat &y)
    {
      return t - y;
    }
  };
}