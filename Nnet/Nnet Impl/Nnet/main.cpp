#include "stdafx.h"
#include "nnet_bpnn.h"


int main(int argc, const char **argv)
{
  mat xy;
  xy.load("../../Data/train.csv");
  double split_factor = 11.0 / 12.0;
  mat x = xy.cols(0, xy.n_cols - 2).t();
  mat y = xy.col(xy.n_cols - 1).t();
  mat tx = x.cols(0, split_factor * x.n_cols - 1);
  mat vx = x.cols(split_factor * x.n_cols, x.n_cols - 1);
  mat ty = y.cols(0, split_factor * y.n_cols - 1);
  mat vy = y.cols(split_factor * y.n_cols, y.n_cols - 1);
  // x.print("x=");
  // y.print("y=");

  nnet::bpnn nn;
  nn.init_malloc({ 23, 23, 20, 1 });

  nn.loss_itf.fill(new nnet::mse);

  nn.alphas(0).fill(0.001); // 0.001, 0.01
  nn.alphas(1).fill(0.002); // 0.002, 0.01
  nn.alphas(2).fill(0.005); // 0.005, 0.01

  nn.lambdas(0).fill(0.1); // 0.1, 0.0
  nn.lambdas(1).fill(0.1); // 0.1, 0.0
  nn.lambdas(2).fill(0.1); // 0.1, 0.0

  nn.biases(0).fill(0.1);
  nn.biases(1).fill(0.1);
  nn.biases(2).fill(0.1);

  nn.weights(0).fill(arma::fill::randn);
  nn.weights(1).fill(arma::fill::randn);
  nn.weights(2).fill(arma::fill::randn);

  nn.layers(0).acts.fill(new nnet::tanh);
  nn.layers(1).acts.fill(new nnet::sigmoid);
  nn.layers(2).acts.fill(new nnet::leaky_relu(0.2));

  nn.layers(0).weight_opt = new nnet::gradient_desc;
  nn.layers(1).weight_opt = new nnet::gradient_desc;
  nn.layers(2).weight_opt = new nnet::gradient_desc;

  nn.layers(0).bias_opt = new nnet::gradient_desc;
  nn.layers(1).bias_opt = new nnet::gradient_desc;
  nn.layers(2).bias_opt = new nnet::gradient_desc;

  mat tlosses;
  mat vlosses;
  int n_iterations = 10000;
  nn.train(
    tx,
    ty,
    n_iterations,
    &tlosses,
    &vx,
    &vy,
    &vlosses
  );
  // nn.ios.print("ios=");
  // nn.muls.print("muls=");
  // nn.deltas.print("deltas=");
  // nn.weights.print("weights=");
  // nn.biases.print("biases=");
  tlosses.save("../../Data/tlosses.csv", arma::csv_ascii);
  vlosses.save("../../Data/vlosses.csv", arma::csv_ascii);
  nn.propagate(vx);
  mat ry = nn.output();
  mat(ry.t()).save("../../Data/predict.csv", arma::csv_ascii);
  mat(vy.t()).save("../../Data/actual.csv", arma::csv_ascii);
  cor(ry, vy).print("corr =");

  return 0;
}