#pragma once
#include "stdafx.h"
#include "nnet_activator.h"
#include "channel.h"

namespace nnet
{
  class unit
  {
  public:
    void bind(
      base::activator *act, 
      subview<double> &in, 
      subview_col<double> &out,
      subview_col<double> &weight,
      subview_col<double> &bias)
    {
      this->act.bind(act);
      this->in.bind(in);
      this->out.bind(out);
      this->weight.bind(weight);
      this->bias.bind(bias);
    }


    void run() { out() = act().activation((weight().t() * in()).t() + bias()); }


    channel<base::activator> act;
    channel<subview<double>> in;
    channel<subview_col<double>> out;
    channel<subview_col<double>> weight;
    channel<subview_col<double>> bias;
  };
}
