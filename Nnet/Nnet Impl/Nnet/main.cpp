#include "stdafx.h"
#include "nnet_nnet.h"


int main(int argc, const char **argv)
{
  mat x({ { 1, 0, 1 },{ 1, 0, 1 } });
  x = x.t();
  mat y({ 1, 1 });
  x.print("x=");
  y.print("y=");

  nnet::loss *mse = new nnet::mse;
  nnet::activation *sigmoid = new nnet::sigmoid;
  nnet::activation *tanh = new nnet::tanh;

  nnet::nnet nn;
  nn.init_malloc({ 3, 2, 1 });
  nn.los = mse;
  nn.alphas(0).fill(0.9);
  nn.alphas(1).fill(0.9);
  nn.lambdas(0).fill(0.0);
  nn.lambdas(1).fill(0.0);
  nn.biases(0) = mat({ -0.4, 0.2 }).t();
  nn.biases(1) = { 0.1 };
  nn.weights(0) =
  {
    { 0.2, 0.4, -0.5 },
    { -0.3, 0.1, 0.2 }
  };
  nn.weights(1) = { -0.3, -0.2 };
  nn.layers(0).acts(0) = sigmoid;
  nn.layers(0).acts(1) = sigmoid;
  nn.layers(1).acts(0) = sigmoid;

  field<mat> losses;
  nn.train(x, y, 1, &losses);
  nn.ios.print("ios=");
  nn.muls.print("muls=");
  nn.deltas.print("deltas=");
  nn.weights.print("weights=");
  nn.biases.print("biases=");
  losses.print("losses=");

  delete sigmoid;
  delete tanh;
  delete mse;

  return 0;
}