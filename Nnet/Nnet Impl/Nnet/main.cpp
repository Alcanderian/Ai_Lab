#include "stdafx.h"
#include "nnet_nnet.h"


int main(int argc, const char **argv)
{
  mat x({ 1, 0, 1 });
  x = x.t();
  mat y({ 1 });
  x.print("x=");
  y.print("y=");

  nnet::nnet nn;
  nn.init_malloc({ 3, 2, 1 });
  nn.los = new nnet::mse;
  nn.alphas(0) = mat({ 0.9, 0.9 }).t();
  nn.alphas(1) = { 0.9 };
  nn.lambdas(0) = mat({ 0, 0 }).t();
  nn.lambdas(1) = { 0 };
  nn.biases(0) = mat({ -0.4, 0.2 }).t();
  nn.biases(1) = { 0.1 };
  nn.weights(0) = 
  { 
    { 0.2, 0.4, -0.5 }, 
    { -0.3, 0.1, 0.2 } 
  };
  nn.weights(1) = { -0.3, -0.2 };
  nn.layers(0).acts(0) = new nnet::sigmoid;
  nn.layers(0).acts(1) = new nnet::sigmoid;
  nn.layers(1).acts(0) = new nnet::sigmoid;

  field<mat> losses;
  nn.train(x, y, 10, &losses);
  nn.conns.print("conns=");
  nn.dlosses.print("dlosses=");
  nn.deltas.print("deltas=");
  nn.bufs.print("bufs=");
  nn.weights.print("weights=");
  nn.biases.print("biases=");
  losses.print("losses=");
  return 0;
}