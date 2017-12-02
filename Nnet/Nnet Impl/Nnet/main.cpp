#include "stdafx.h"
#include "nnet_nnet.h"


int main(int argc, const char **argv)
{
  mat xy;
  xy.load("../../Data/fix_train.csv");
  mat x = xy.cols(0, xy.n_cols - 2).t();
  mat y = xy.col(xy.n_cols - 1).t() / 100;
  // x.print("x=");
  // y.print("y=");

  nnet::loss *mse = new nnet::mse;
  nnet::activation *sigmoid = new nnet::sigmoid;
  nnet::activation *tanh = new nnet::tanh;
  nnet::activation *prelu = new nnet::para_relu(0.5);

  nnet::nnet nn;
  nn.init_malloc({ 11, 5, 1 });
  nn.loss_itf.fill(mse);
  nn.alphas(0).fill(0.1);
  nn.alphas(1).fill(0.1);
  nn.lambdas(0).fill(0.0);
  nn.lambdas(1).fill(0.0);
  nn.biases(0).fill(0.1);
  nn.biases(1).fill(0.1);
  nn.weights(0).fill(0.1);
  nn.weights(1).fill(0.1);
  nn.layers(0).acts.fill(sigmoid);
  nn.layers(1).acts.fill(prelu);

  field<mat> losses;
  nn.train(x, y, 100, &losses);
  // nn.ios.print("ios=");
  // nn.muls.print("muls=");
  // nn.deltas.print("deltas=");
  nn.weights.print("weights=");
  nn.biases.print("biases=");
  // losses.print("losses=");

  delete sigmoid;
  delete tanh;
  delete mse;

  return 0;
}