#pragma once
#include "stdafx.h"


namespace nnet
{
    class activation 
    {
    public:
      // propagate(activation) function
      virtual void propagate(const mat &z, mat *y) = 0;
      // back_propagate(derivation) of activation function
      virtual void back_propagate(const mat &z, mat *y) = 0;
    };


    class sigmoid :
      public activation 
    {
    public:
      // 1 / (1 + e^z)
      void propagate(const mat &z, mat *y) { *y =  1.0 / (1.0 + exp(-z)); }
      // z / (1 - z)
      void back_propagate(const mat &z, mat *y) { *y = z % (1.0 - z); }
    };


    class tanh :
      public activation 
    {
    public:
      // (e^z - e^(-z)) / (e^z + e^(-z)),
      void propagate(const mat &z, mat *y) { *y =  arma::tanh(z); }
      // 1 - z^2
      void back_propagate(const mat &z, mat *y) { *y =  1.0 - square(z); }
    };


    class leaky_relu :
      public activation 
    {
    public:
      double beta;
      leaky_relu(const double &beta = 0.0) :
        beta(beta)
      { }


      void propagate(const mat &z, mat *y) { assert(beta >= 0.0); *y = z; y->elem(find(*y < 0.0)) *= beta; }
      void back_propagate(const mat &z, mat *y) { *y = ones(z.n_rows, z.n_cols); y->elem(find(z < 0.0)).fill(beta); }
    };


    class identity :
      public activation
    {
    public:
      void propagate(const mat &z, mat *y) { *y = z; }
      void back_propagate(const mat &z, mat *y) { *y = ones(z.n_rows, z.n_cols); }
    };
}
