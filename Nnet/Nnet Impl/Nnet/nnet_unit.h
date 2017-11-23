#pragma once
#include "stdafx.h"
#include "nnet_activator.h"
#include "channel.h"

namespace nnet
{
  class unit
  {
  public:
    void bind(base::activator *act, vec &in, double &out, vec &weight, double &bias)
    {
      this->act.bind(act);
      this->in.bind(in);
      this->out.bind(out);
      this->weight.bind(weight);
      this->bias.bind(bias);
    }


    void run() { out() = dot(weight(), in()) + bias(); }


    channel<base::activator> act;
    channel<vec> in;
    channel<double> out;
    channel<vec> weight;
    channel<double> bias;
  };
}
