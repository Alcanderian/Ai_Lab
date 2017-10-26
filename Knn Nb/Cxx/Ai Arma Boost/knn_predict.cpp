#include <armadillo>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <boost/algorithm/string.hpp>

#include "knn_predict.h"

using namespace std;

void knn_predict::build_tst_vec(
  const vector<string> &tst_pattern, arma::vec &tst)
{
  int actual_size, unknown_cnt = 0;

  tst = arma::vec(words).zeros();
  if (knn_cfg.mat_type == knn_config::mat_onehot)
    for (int i = 0; i < tst_pattern.size(); ++i)
      if (word_idx.find(tst_pattern[i]) != word_idx.end())
        tst(word_idx[tst_pattern[i]]) = 1; // Ignore unknown word.

  if (knn_cfg.mat_type >= knn_config::mat_tfidf)
  {
    for (int i = 0; i < tst_pattern.size(); ++i) // Get t.
    {
      if (word_idx.find(tst_pattern[i]) != word_idx.end())
        tst(word_idx[tst_pattern[i]])++; // Ignore unknown word.
      else ++unknown_cnt; // Count unknow word.
    }
    actual_size = tst_pattern.size() - unknown_cnt;
    if (actual_size != 0)
      for (int i = 0; i < words; ++i)
        if (tst(i) != 0)
        {
          tst(i) /= actual_size; // Get tf.
          if (knn_cfg.mat_type == knn_config::mat_tfidf)
            tst(i) *= idf(i); // Get tfidf.
        }
  }
  if (data_gen_cfg.vector_type == data_gen_config::normalised_vector)
    tst = arma::normalise(tst);
}

void knn_predict::get_tst_distance(
  const arma::vec &tst, arma::vec &distance)
{
  /*
  Note: norm(A - B, p) -> p-norm(A - B)
        norm_dot(A, B) -> cosin(A, B).
  */
  distance = arma::vec(patterns);
  for (int i = 0; i < patterns; ++i)
  {
    if (knn_cfg.mat_type == knn_config::mat_onehot)
      distance(i) = (knn_cfg.distance_type == knn_config::norm_distance)
      ? arma::norm(tst - onehot.col(i), knn_cfg.norm_order)
      : arma::norm_dot(tst, onehot.col(i));
    else if (knn_cfg.mat_type == knn_config::mat_tfidf)
      distance(i) = (knn_cfg.distance_type == knn_config::norm_distance)
      ? arma::norm(tst - tfidf.col(i), knn_cfg.norm_order)
      : arma::norm_dot(tst, tfidf.col(i));
    else
      distance(i) = (knn_cfg.distance_type == knn_config::norm_distance)
      ? arma::norm(tst - tf.col(i), knn_cfg.norm_order)
      : arma::norm_dot(tst, tf.col(i));
  }
}

/*
Fix topK.
For p-norm distance, it is topK.
For cosin distance, it is reverse topK.
*/
void knn_predict::get_top_k_range(const int &top_k, int &start, int &end)
{
  if (knn_cfg.distance_type == knn_config::norm_distance)
  {
    start = 0, end = top_k;
    if (end > patterns) end = patterns;
  }
  else
  {
    start = patterns - top_k, end = patterns;
    if (start < 0) start = 0;
  }
}

knn_result knn_predict::predict_one(
  const vector<string> &tst_pattern, const int &top_k)
{
  arma::vec tst, zero_vec;
  int top_k_start, top_k_end;
  arma::uvec idx;
  knn_result ret;

  ret.prob = arma::vec(emoj_name.size()).zeros();
  zero_vec = arma::vec(emoj_name.size()).zeros();
  build_tst_vec(tst_pattern, tst);
  get_tst_distance(tst, ret.distance);
  get_top_k_range(top_k, top_k_start, top_k_end);
  ret.top_k = top_k_end - top_k_start;

  // [Fuck]Merge sort or quick sort? Which is better? Determined by the test set, not by train set.
  idx = arma::stable_sort_index(ret.distance);

  if (knn_cfg.predict_type == knn_config::predict_with_dist_weight)
  {
    if (knn_cfg.distance_type == knn_config::norm_distance)
      // Let distance = 1e+10 to prevent from div_zero.
      ret.distance.for_each([](double &n) { n = n == 0 ? 1e+10 : 1 / n; });
    else
      // When distance == 1 in cosin distance, that is myself, set weight to 1e+10. 
      ret.distance.for_each([](double &n) { if (n == 1) n = 1e+10; });
  }

  for (int i = top_k_start; i < top_k_end; ++i)
  {
    if (knn_cfg.predict_type == knn_config::predict_with_dist_weight)
      ret.prob += emoj_prob.col(idx(i)) * pow(ret.distance(idx(i)), knn_cfg.weight_type);
    else
      ret.prob += emoj_prob.col(idx(i));
  }

  if (knn_cfg.predict_type == knn_config::predict_with_dist_weight)
  {
    // If all prob is zero, set it to 1 / count(emotion).
    if (arma::sum(ret.prob != zero_vec) == 0)
      ret.prob.fill(double(1) / emoj_name.size());
    ret.prob = arma::normalise(ret.prob, 1);
  }
  ret.label = emoj_name[ret.prob.index_max()];

  return ret;
}

void knn_predict::classification_verify(
  const string &veri_csv, const string &result_csv, const int &top_k)
{
  int pos = 0, neg = 0, top_n;
  ifstream f_veri(veri_csv);
  ofstream f_result(result_csv);
  vector<string> v_buf;
  knn_result k_buf;
  string s1_buf, s2_buf;

  top_n = top_k < 0 ? (int)sqrt(patterns) : top_k;

  getline(f_veri, s1_buf);
  f_result << "Words (split by space),actual label,classified label\n";
  while (getline(f_veri, s1_buf))
  {
    boost::split(v_buf, s1_buf, boost::is_any_of(","));
    f_result << s1_buf;

    s1_buf = v_buf[0], s2_buf = v_buf[1];
    boost::split(v_buf, s1_buf, boost::is_any_of(" "));
    k_buf = predict_one(v_buf, top_n);

    f_result << "," << k_buf.label << endl;
    if (k_buf.label == s2_buf) ++pos;
    else ++neg;
  }

  cout
    << "top_k\t" << top_n << endl
    << "distance_type\t";
  if (knn_cfg.distance_type == knn_config::norm_distance)
    cout << knn_cfg.norm_order << "-norm" << endl;
  else
    cout << "cosin" << endl;
  cout << "vector_type\t";
  if (data_gen_cfg.vector_type == data_gen_config::normalised_vector)
    cout << "normalised" << endl;
  else
    cout << "raw" << endl;
  cout << "mat_type\t";
  if (knn_cfg.mat_type == knn_config::mat_onehot)
    cout << "onehot" << endl;
  else if (knn_cfg.mat_type == knn_config::mat_tfidf)
    cout << "tfidf" << endl;
  else
    cout << "tf" << endl;
  cout
    << "correct/total\t" << pos << "/" << pos + neg << endl
    << "accuracy\t" << double(pos) / (pos + neg) << endl;

  f_result
    << "top_k," << top_n << endl
    << "distance_type,";
  if (knn_cfg.distance_type == knn_config::norm_distance)
    f_result << knn_cfg.norm_order << "-norm" << endl;
  else
    f_result << "cosin" << endl;
  f_result << "vector_type,";
  if (data_gen_cfg.vector_type == data_gen_config::normalised_vector)
    f_result << "normalised" << endl;
  else
    f_result << "raw" << endl;
  f_result << "mat_type,";
  if (knn_cfg.mat_type == knn_config::mat_onehot)
    f_result << "onehot" << endl;
  else if (knn_cfg.mat_type == knn_config::mat_tfidf)
    f_result << "tfidf" << endl;
  else
    f_result << "tf" << endl;
  f_result
    << "correct/total," << pos << "/" << pos + neg << endl
    << "accuracy," << double(pos) / (pos + neg) << endl;

  f_veri.close(), f_result.close();
}

void knn_predict::classification(
  const string &tst_csv, const string &result_csv, const int &top_k)
{
  ifstream f_tst(tst_csv);
  ofstream f_result(result_csv);
  vector<string> v_buf;
  knn_result k_buf;
  string s_buf;
  int top_n;

  top_n = top_k < 0 ? (int)sqrt(patterns) : top_k;
  top_n = top_n > patterns ? patterns : top_n;

  getline(f_tst, s_buf);
  f_result << "textid,label\n";
  while (getline(f_tst, s_buf))
  {
    boost::split(v_buf, s_buf, boost::is_any_of(","));
    f_result << v_buf[0] << ",";

    s_buf = v_buf[1];
    boost::split(v_buf, s_buf, boost::is_any_of(" "));
    k_buf = predict_one(v_buf, top_n);

    f_result << k_buf.label << endl;
  }

  f_tst.close(), f_result.close();
}

void knn_predict::regression_verify(
  const string &veri_csv, const string &result_csv, const int &top_k)
{
  if (knn_cfg.predict_type != knn_config::predict_with_dist_weight)
    throw exception("Regression must use knn_config::predict_with_dist_weight.");

  arma::mat veri_prob, result_prob, cor_prob;
  vector<string> veri_buf, v_buf;
  vector<arma::vec> result_buf;
  ifstream f_veri(veri_csv);
  ofstream f_result(result_csv);
  stringstream ss_buf;
  knn_result k_buf;
  string s_buf;
  int top_n;

  top_n = top_k < 0 ? (int)sqrt(patterns) : top_k;
  top_n = top_n > patterns ? patterns : top_n;

  getline(f_veri, s_buf);
  f_result << "anger,disgust,fear,joy,sad,surprise\n";
  while (getline(f_veri, s_buf))
  {
    boost::replace_first(s_buf, ",", "\t"); // "x,y,z..." -> "x\ty,z...".
    boost::replace_all(s_buf, ",", " "); // "x\ty,z..." -> "x\ty z...".
    boost::split(v_buf, s_buf, boost::is_any_of("\t")); // Get "x", "y z...".

    s_buf = v_buf[0];
    veri_buf.push_back(v_buf[1]);
    boost::split(v_buf, s_buf, boost::is_any_of(" "));
    k_buf = predict_one(v_buf, top_n);

    for (int i = 0; i < emoj_name.size(); ++i)
      f_result << k_buf.prob(i) << (i == int(emoj_name.size()) - 1 ? "\n" : ",");
    result_buf.push_back(k_buf.prob);
  }

  veri_prob = arma::mat(veri_buf.size(), emoj_name.size());
  result_prob = arma::mat(result_buf.size(), emoj_name.size());

  for (int i = 0; i < veri_buf.size(); ++i)
  {
    ss_buf.clear();
    ss_buf.str(veri_buf[i]);
    for (int j = 0; j < emoj_name.size(); ++j)
      ss_buf >> veri_prob(i, j);
  }

  for (int i = 0; i < result_buf.size(); ++i)
    for (int j = 0; j < emoj_name.size(); ++j)
      result_prob(i, j) = result_buf[i](j);

  /*
  This function returns the correlation matrix.
  Note: c = cor(X, Y),
        c(i, j) is the correlation of X.col(i) and Y.col(j).
  */
  cor_prob = arma::cor(veri_prob, result_prob);

  cout
    << "top_k\t" << top_n << endl
    << "distance_type\t";
  if (knn_cfg.distance_type == knn_config::norm_distance)
    cout << knn_cfg.norm_order << "-norm" << endl;
  else
    cout << "cosin" << endl;
  cout << "vector_type\t";
  if (data_gen_cfg.vector_type == data_gen_config::normalised_vector)
    cout << "normalised" << endl;
  else
    cout << "raw" << endl;
  cout << "mat_type\t";
  if (knn_cfg.mat_type == knn_config::mat_onehot)
    cout << "onehot" << endl;
  else if (knn_cfg.mat_type == knn_config::mat_tfidf)
    cout << "tfidf" << endl;
  else
    cout << "tf" << endl;
  cout << "correls\t";
  for (int i = 0; i < emoj_name.size(); ++i)
    cout << cor_prob(i, i) << (i == int(emoj_name.size()) - 1 ? "\n" : ",");
  cout << "avg_correl\t" << arma::trace(cor_prob) / emoj_name.size() << endl; // trace(A) -> sum(A.diag()).
  
  f_result
    << "top_k," << top_n << endl
    << "distance_type,";
  if (knn_cfg.distance_type == knn_config::norm_distance)
    f_result << knn_cfg.norm_order << "-norm" << endl;
  else
    f_result << "cosin" << endl;
  f_result << "vector_type,";
  if (data_gen_cfg.vector_type == data_gen_config::normalised_vector)
    f_result << "normalised" << endl;
  else
    f_result << "raw" << endl;
  f_result << "mat_type,";
  if (knn_cfg.mat_type == knn_config::mat_onehot)
    f_result << "onehot" << endl;
  else if (knn_cfg.mat_type == knn_config::mat_tfidf)
    f_result << "tfidf" << endl;
  else
    f_result << "tf" << endl;
  f_result << "correls,";
  for (int i = 0; i < emoj_name.size(); ++i)
    f_result << cor_prob(i, i) << (i == int(emoj_name.size()) - 1 ? "\n" : ",");
  f_result << "avg_correl," << arma::trace(cor_prob) / emoj_name.size() << endl; // trace(A) -> sum(A.diag()).

  f_veri.close(), f_result.close();
}

void knn_predict::regression(
  const string &tst_csv, const string &result_csv, const int &top_k)
{
  if (knn_cfg.predict_type != knn_config::predict_with_dist_weight)
    throw exception("Regression must use knn_config::predict_with_dist_weight.");

  ifstream f_tst(tst_csv);
  ofstream f_result(result_csv);
  vector<string> v_buf;
  knn_result k_buf;
  string s_buf;
  int top_n;

  top_n = top_k < 0 ? (int)sqrt(patterns) : top_k;
  top_n = top_n > patterns ? patterns : top_n;

  getline(f_tst, s_buf);
  boost::replace_first(s_buf, ",", "\t");
  boost::replace_first(s_buf, ",", "\t");
  boost::split(v_buf, s_buf, boost::is_any_of("\t"));
  f_result << v_buf[0] << "," << v_buf[2] << endl;
  while (getline(f_tst, s_buf))
  {
    boost::split(v_buf, s_buf, boost::is_any_of(","));
    f_result << v_buf[0];

    s_buf = v_buf[1];
    boost::split(v_buf, s_buf, boost::is_any_of(" "));
    k_buf = predict_one(v_buf, top_n);

    for (int i = 0; i < emoj_name.size(); ++i)
      f_result << "," << k_buf.prob(i);
    f_result << endl;
  }

  f_tst.close(), f_result.close();
}
