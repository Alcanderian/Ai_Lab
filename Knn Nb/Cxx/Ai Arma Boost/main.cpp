#include <armadillo>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <boost/algorithm/string.hpp>

#include "data_gen.h"
#include "knn_predict.h"
#include "nb_predict.h"
#include "fix_classi_csv.h"

using namespace std;

enum main_option
{
  start_of_opt = 0,
  knn_export_mat,
  knn_classification_verify,
  knn_regression_verify,
  knn_classification,
  knn_regression,
  nb_export_mat,
  nb_classification_verify,
  nb_regression_verify,
  nb_classification,
  nb_regression,
  end_of_opt
} opts;

int main(int argc, const char **argv)
{
  knn_predict knn;
  nb_predict nb;

  ios::sync_with_stdio(false);

  // Fix csv format.
  fix_classi_csv(
    "classification_train_set.csv",
    "classification_fixed_train_set.csv");

  cout
    << "0. exit" << endl
    << "1. knn_export_mat" << endl
    << "2. knn_classification_verify" << endl
    << "3. knn_regression_verify" << endl
    << "4. knn_classification" << endl
    << "5. knn_regression" << endl
    << "6. nb_export_mat" << endl
    << "7. nb_classification_verify" << endl
    << "8. nb_regression_verify" << endl
    << "9. nb_classification" << endl
    << "10. nb_regression" << endl
    << endl
    << "input option: ";

  int opt, topk;
  while (cin >> opt)
  {
    if (opt == main_option::start_of_opt)
      break;
    if (!(opt > main_option::start_of_opt && opt < main_option::end_of_opt))
    {
      cout << "input option: ";
      continue;
    }

    cout << endl;

    switch (opt)
    {
    case knn_classification_verify:
    case knn_classification:
      knn.data_gen_cfg.vector_type = data_gen_config::normalised_vector;
      knn.knn_cfg.distance_type = knn_config::norm_distance;
      knn.knn_cfg.mat_type = knn_config::mat_tfidf;
      knn.knn_cfg.predict_type = knn_config::predict_with_dist_weight;
      knn.knn_cfg.weight_type = knn_config::weight_linear;
      knn.knn_cfg.norm_order = 2;
      topk = 11;
      knn.import_csv("classification_fixed_train_set.csv");
      break;
    case knn_regression_verify:
    case knn_regression:
      knn.data_gen_cfg.vector_type = data_gen_config::normalised_vector;
      knn.knn_cfg.distance_type = knn_config::cos_distance;
      knn.knn_cfg.mat_type = knn_config::mat_tfidf;
      knn.knn_cfg.predict_type = knn_config::predict_with_dist_weight;
      knn.knn_cfg.weight_type = knn_config::weight_square;
      topk = 11;
      knn.import_csv("regression_train_set.csv");
      break;
    case nb_classification_verify:
    case nb_classification:
      nb.data_gen_cfg.vector_type = data_gen_config::raw_vector;
      nb.nb_cfg.laplace_alpha = 0.595;
      nb.import_csv("classification_fixed_train_set.csv");
      break;
    case nb_regression_verify:
    case nb_regression:
      nb.data_gen_cfg.vector_type = data_gen_config::raw_vector;
      nb.nb_cfg.laplace_alpha = 0.035;
      nb.import_csv("regression_train_set.csv");
      break;
    default:
      break;
    }

    switch (opt)
    {
    case knn_export_mat:
      knn.export_mat("onehot.mat", "t.mat", "tf.mat", "df.mat", "idf.mat", "tfidf.mat", "emoj_prob.mat");
      break;
    case knn_classification_verify:
      knn.classification_verify(
        "classification_validation_set.csv",
        "classification_verify_result.csv",
        topk);
      break;
    case knn_regression_verify:
      knn.regression_verify(
        "regression_validation_set.csv",
        "regression_verify_result.csv",
        topk);
      break;
    case knn_classification:
      knn.classification(
        "classification_test_set.csv",
        "classification_test_result.csv",
        topk);
      cout << "done." << endl;
      break;
    case knn_regression:
      knn.regression(
        "regression_test_set.csv",
        "regression_test_result.csv",
        topk);
      cout << "done." << endl;
      break;
    case nb_export_mat:
      nb.export_mat("onehot.mat", "t.mat", "tf.mat", "df.mat", "idf.mat", "tfidf.mat", "emoj_prob.mat");
      nb.export_nbmat("emoj_word_cnt.mat", "word_emoj_cnt.mat", "prob_emoj.mat", "patterns_size.mat");
      break;
    case nb_classification_verify:
      nb.classification_verify(
        "classification_validation_set.csv",
        "classification_verify_result.csv");
      break;
    case nb_regression_verify:
      nb.regression_verify(
        "regression_validation_set.csv",
        "regression_verify_result.csv");
      break;
    case nb_classification:
      nb.classification(
        "classification_test_set.csv",
        "classification_test_result.csv");
      cout << "done." << endl;
      break;
    case nb_regression:
      nb.regression(
        "regression_test_set.csv",
        "regression_test_result.csv");
      cout << "done." << endl;
      break;
    default:
      break;
    }

    cout << endl;
    cout << "input option: ";
  }

  return 0;
}
