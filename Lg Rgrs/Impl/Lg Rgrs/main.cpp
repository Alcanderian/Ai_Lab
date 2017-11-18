#include "logit_model.hpp"
#include "evaluation.hpp"

int main(int argc, const char **argv)
{
  std::srand((uint32_t)time(0));

  mat Xy;
  Xy.load("../../Data/train.csv");
  mat tXy = Xy(span(0, Xy.n_rows * 0.75 - 1), span::all);
  mat vXy = Xy(span(Xy.n_rows * 0.75, Xy.n_rows - 1), span::all);
  mat vX = vXy(span::all, span(0, vXy.n_cols - 2));
  vec vy = vXy.col(vXy.n_cols - 1);

  lr_model lr;
  lr.set_data(tXy);
  lr.set_cfg({
    0.01, // alpha
    600, // lambda
    0.001, // eps
    lr_config::constant_rate,
    lr_config::full_descent,
    lr_config::ones_weight
  });

  vec w = lr.train(500);
  w.raw_print("final w =");
  vec r = lr.classification(vX, w);
  auto e = evaluation(vy, r);
  for (auto k : e) {
    std::cout << k.first << ":" << k.second << std::endl;
  }
  system("pause");

  return 0;
}