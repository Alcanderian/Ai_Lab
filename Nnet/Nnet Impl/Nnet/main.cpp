#include "stdafx.h"
#include "nnet_bpnn.h"
#include <armadillo>

int main(int argc, const char **argv)
{
  mat xy, sx;
  xy.load("../../../Fi Project/Data/bc/urain.csv");
  sx.load("../../../Fi Project/Data/bc/uest.csv");
  sx = sx.t();
  double split_factor = 5.1 / 10.0;
  mat x = xy.cols(0, xy.n_cols - 2).t();
  mat y = xy.col(xy.n_cols - 1).t();

  int n_train_samples = x.n_cols;
  mat xx = join_horiz(x, sx);
  for (int i = 0; i < xx.n_rows; ++i) {
    xx.row(i) = (xx.row(i) - min(xx.row(i))) / (max(xx.row(i)) - min(xx.row(i)));
  }
  x = xx.cols(0, n_train_samples - 1);
  sx = xx.cols(n_train_samples, xx.n_cols - 1);
  
  mat tx = x.cols(0, split_factor * x.n_cols - 1);
  mat vx = x.cols(split_factor * x.n_cols, x.n_cols - 1);
  mat ty = y.cols(0, split_factor * y.n_cols - 1);
  mat vy = y.cols(split_factor * y.n_cols, y.n_cols - 1);
  // x.print("x=");
  // y.print("y=");

  nnet::bpnn nn;
  nn.init_malloc({ 16, 16, 16, 1 });

  nn.loss_itfs.fill(new nnet::xent);

  nn.alphas(0).fill(0.01);
  nn.alphas(1).fill(0.01);
  nn.alphas(2).fill(0.01);

  nn.lambdas(0).fill(0.0);
  nn.lambdas(1).fill(0.0);
  nn.lambdas(2).fill(0.0);

  nn.biases(0).fill(arma::fill::randn);
  nn.biases(1).fill(arma::fill::randn);
  nn.biases(2).fill(arma::fill::randn);

  nn.weights(0).fill(arma::fill::randn);
  nn.weights(1).fill(arma::fill::randn);
  nn.weights(2).fill(arma::fill::randn);

  nn.layers(0).act = new nnet::identity;
  nn.layers(1).act = new nnet::sigmoid;
  nn.layers(2).act = new nnet::sigmoid;

  nn.layers(0).weight_opt = new nnet::adam;
  nn.layers(1).weight_opt = new nnet::adam;
  nn.layers(2).weight_opt = new nnet::adam;

  nn.layers(0).bias_opt = new nnet::adam;
  nn.layers(1).bias_opt = new nnet::adam;
  nn.layers(2).bias_opt = new nnet::adam;

  mat tlosses;
  mat vlosses;
  int n_iterations = 10;
  nn.train(
    tx,
    ty,
    n_iterations,
    &tlosses,
    &vx,
    &vy,
    &vlosses
  );
  nn.propagate(vx);
  nnet::loss *f1 = new nnet::nf1;
  mat eval;
  f1->avg_eval(nn.output(), vy, &eval);
  (-eval).print("f1 =");

  nn.propagate(sx);
  mat syp = nn.output();
  syp.elem(find(syp < 0.5)).fill(0.0);
  syp.elem(find(syp >= 0.5)).fill(1.0);
  arma::imat sy(syp.n_rows, syp.n_cols);
  int i = 0;
  sy.for_each([&i, &syp](int64_t &n) { n = syp(i++); });
  arma::imat(sy.t()).save("../../../Fi Project/Data/bc/uesult.csv", arma::csv_ascii);
  return 0;
}