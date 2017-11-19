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
  Xy.load("../../Data/train.arm", arma::arma_binary);
  auto tXy = Xy.rows(0, Xy.n_rows * 0.75 - 1);
  auto vXy = Xy.rows(Xy.n_rows * 0.75, Xy.n_rows - 1);
  auto vX = vXy.cols(0, vXy.n_cols - 2);
  auto vy = vXy.col(vXy.n_cols - 1);

  lr_model lr;
  lr.set_data(tXy);
  lr.set_cfg({
    0.01, // alpha
    600, // lambda
    0.0001, // eps
    lr_config::rate::constant_rate,
    lr_config::descent::full_descent,
    lr_config::weight::ones_weight
  });

  vec w = lr.train(200);
  w.raw_print("final w =");
  vec re = lr.classification(vX, w);
  for (auto e : evaluation(vy, re))
    cout << e.first << ":" << e.second << endl;
  system("pause");

  return 0;
}