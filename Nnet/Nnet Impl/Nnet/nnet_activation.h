#pragma once
#include "stdafx.h"


namespace nnet
{
    class activation 
    {
    public:
      // propagate(activation) function
      virtual mat propagate(const mat &z) = 0;
      // back_propagate(derivation) of activation function
      virtual mat back_propagate(const mat &z) = 0;
    };


    class sigmoid :
      public activation 
    {
    public:
      // 1 / (1 + e^z)
      mat propagate(const mat &z) { return 1.0 / (1.0 + exp(-z)); }
      // s / (1 - s)
      mat back_propagate(const mat &z) { mat &s = propagate(z); return s % (1.0 - s); }
    };


    class tanh :
      public activation 
    {
    public:
      // (e^z - e^(-z)) / (e^z + e^(-z)),
      mat propagate(const mat &z) { return arma::tanh(z); }
      // 1 - t^2
      mat back_propagate(const mat &z) { mat t = arma::tanh(z); return 1 - square(t); }
    };


    class leaky_relu :
      public activation 
    {
    public:
      double beta;
      leaky_relu(const double &beta = 0.0) :
        beta(beta)
      { }


      mat propagate(const mat &z) { mat r = z; r.elem(find(r < 0.0)) *= beta; return r; }
      mat back_propagate(const mat &z) { mat r = ones(z.n_rows, z.n_cols); r.elem(find(z < 0.0)).fill(beta); return r; }
    };


    class identity :
      public activation
    {
    public:
      mat propagate(const mat &z) { return z; }
      mat back_propagate(const mat &z) { return ones(z.n_rows, z.n_cols); }
    };
}
