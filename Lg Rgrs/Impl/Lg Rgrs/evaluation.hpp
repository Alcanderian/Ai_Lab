#pragma once
#include <armadillo>
#include <string>
#include <map>

using std::string;
using std::map;
using arma::vec;
using arma::uvec;

/* tag actual predict state(actual + 2 * predict)
   tp   1      1       3
   fn   1      0       1
   tn   0      0       0
   fp   0      1       2 */
map<string, double> evaluation(const vec &actual, const vec &predict,
  const double &pos = 1.0, const double &neg = 0.0) {
  /* tranfrom to uniform state */
  uvec ua(actual.n_elem), up(predict.n_elem);
  ua.elem(find(actual == pos)).fill(1);
  ua.elem(find(actual == neg)).fill(0);
  up.elem(find(predict == pos)).fill(1);
  up.elem(find(predict == neg)).fill(0);

  uvec state = ua + 2 * up;
  double tp = uvec(find(state == 3)).n_elem;
  double fn = uvec(find(state == 1)).n_elem;
  double tn = uvec(find(state == 0)).n_elem;
  double fp = uvec(find(state == 2)).n_elem;

  double accuracy = (tp + tn) / (tp + fp + tn + fn);
  double recall = tp / (tp + fn);
  double precision = tp / (tp + fp);
  double f1 = 2 * precision * recall / (precision + recall);

  map<string, double> evaluation;
  evaluation["accuracy"] = accuracy;
  evaluation["recall"] = recall;
  evaluation["precision"] = precision;
  evaluation["f1"] = f1;

  return evaluation;
}
