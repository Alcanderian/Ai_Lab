#pragma once

#include <armadillo>
#include <vector>
#include <string>

#include "data_gen.h"

using namespace std;

struct knn_config
{
  enum distance_t
  {
    cos_distance,
    norm_distance
  } distance_type = norm_distance;

  enum mat_t
  {
    mat_onehot,
    mat_tfidf,
    mat_tf
  } mat_type = mat_onehot;

  enum predict_t
  {
    predict_with_dist_weight,
    predict_without_dist_weight
  } predict_type = predict_without_dist_weight;

  enum weight_t
  {
    weight_linear = 1,
    weight_square
  } weight_type = weight_linear;

  int norm_order = 2;
};

struct knn_result
{
  arma::vec prob, distance;
  string label;
  int top_k;
};

struct knn_predict : data_gen
{
  knn_config knn_cfg;

  void build_tst_vec(const vector<string> &tst_pattern, arma::vec &tst);
  void get_tst_distance(
    const arma::vec &tst, arma::vec &distance);
  void get_top_k_range(const int &top_k, int &start, int &end);
  knn_result predict_one(const vector<string> &tst_pattern, const int &top_k);

  void classification_verify(
    const string &veri_csv, const string &result_csv, const int &top_k = -1);
  void classification(
    const string &tst_csv, const string &result_csv, const int &top_k = -1);
  void regression_verify(
    const string &veri_csv, const string &result_csv, const int &top_k = -1);
  void regression(
    const string &tst_csv, const string &result_csv, const int &top_k = -1);
};
