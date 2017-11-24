#include "logit_model.hpp"
#include "evaluation.hpp"

using std::cout;
using std::endl;

void csv_to_binary(const string &prefix) {
  mat t;
  t.load("../../Data/" + prefix + "train.csv", arma::csv_ascii);
  t.save("../../Data/" + prefix + "train.arm", arma::arma_binary);
  t.load("../../Data/" + prefix + "test.csv", arma::csv_ascii);
  t.save("../../Data/" + prefix + "test.arm", arma::arma_binary);
}

int main(int argc, const char **argv)
{
  mat Xy;
  Xy.load("../../Data/ystrain.txt");
  Xy.print("train = ");
  auto X = Xy.cols(0, Xy.n_cols - 2);
  auto y = Xy.col(Xy.n_cols - 1);
  auto tXy = Xy.rows(0, Xy.n_rows * 0.75 - 1);
  auto vXy = Xy.rows(Xy.n_rows * 0.75, Xy.n_rows - 1);
  auto vX = vXy.cols(0, vXy.n_cols - 2);
  auto vy = vXy.col(vXy.n_cols - 1);

  lr_config sgd =
  {
    0.001, // eta
    0.008, // lambda
    0.00001, // eps
    lr_config::rate::constant_rate,
    lr_config::descent::stochastic_descent,
    lr_config::weight::ones_weight
  }; // iteration = 60000

  lr_config gd =
  {
    0.2, // eta
    800.0, // lambda
    0.001, // eps
    lr_config::rate::error_rate,
    lr_config::descent::full_descent,
    lr_config::weight::ones_weight
  }; // iteration = 400

  lr_config best =
  {
    0.1, // eta
    0.0, // lambda
    0.0, // eps
    lr_config::rate::constant_rate,
    lr_config::descent::full_descent,
    lr_config::weight::ones_weight
  }; // iteration = 20000

  lr_config test =
  {
    5.0, // eta
    0.0, // lambda
    0.0, // eps
    lr_config::rate::constant_rate,
    lr_config::descent::full_descent,
    lr_config::weight::zeros_weight
  };

  lr_model lr;
  lr.set_data(Xy);
  lr.set_cfg(test);

  vec cost;
  vec w = lr.train(1, cost);
  w.print("w = ");
  mat sX;
  sX.load("../../Data/ystest.txt");
  sX.print("test =");
  vec re = lr.classification(sX.cols(0, sX.n_cols - 2), w);
  re.print("result = ");
  return 0;
}