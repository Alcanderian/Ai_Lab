#pragma once
#include "stdafx.h"

namespace nnet
{
  namespace base
  {
    class activator {
    public:
      /* activation function */
      virtual vec activation(const vec &z) = 0;
      /* derivation of activation function */
      virtual vec derivation(const vec &z) = 0;
    };
  }

  namespace activator
  {
    class sigmoid :
      public base::activator {
    public:
      vec activation(const vec &z) { return 1.0 / (1.0 + exp(-z)); }
      vec derivation(const vec &z) { vec &s = activation(z); return s % (1.0 - s); }
    };


    class tanh :
      base::activator {
    public:
      vec activation(const vec &z) { return arma::tanh(z); }
      vec derivation(const vec &z) { vec e = exp(2.0 * z); return (e - 1.0) / (e + 1.0); }
    };


    class relu :
      base::activator {
    public:
      vec activation(const vec &z) { vec r = z; r.elem(find(r < 0.0)).fill(0.0); return r; }
      vec derivation(const vec &z) { vec r = zeros(z.n_elem); r.elem(find(z >= 0.0)).fill(1.0); return r; }
    };
  }
}
