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

  mat m = zeros(2, 2);
  m.print();
  subview_col<double> sv = m.col(0);
  sv = v;
  m.print();
  m(0, 0) = 5;
  m.print();
  v.print();
  return 0;
}