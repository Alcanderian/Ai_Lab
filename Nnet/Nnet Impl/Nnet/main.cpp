#include "stdafx.h"
#include "channel.h"
#include "nnet_activator.h"

int main(int argc, const char **argv)
{
  vec v({ 1, 2 });
  channel<double> c(v(1));
  channel<double> d = c;
  d() = 3;
  v.print();

  channel<nnet::base::activator> act(new nnet::activator::sigmoid);
  act().activation(v).print();
  return 0;
}