#pragma once

#include <armadillo>
#include <unordered_map>
#include <string>
#include <vector>

using namespace std;

struct data_gen_config 
{
  enum vector_t
  {
    raw_vector,
    normalised_vector
  } vector_type = raw_vector;
};

struct data_gen
{
  vector<vector<string>> pattern_set;
  vector<string> emoj_name;
  unordered_map<string, int> word_idx;

  // [!]Pattern vector is declared as colvec.
  arma::mat onehot;
  arma::mat t;
  arma::mat tf;
  arma::mat tfidf;
  arma::mat emoj_prob;
  arma::vec df;
  arma::vec idf;

  // [!]Must set before import.
  data_gen_config data_gen_cfg;

  int rows = 0, cols = 0;
  const int &patterns = cols, &words = rows;

  virtual void import_csv(const string &src_csv);
  virtual ~data_gen() = default;

  void export_mat(const string &onehot_f, const string &t_f, const string &tf_f,
    const string &df_f, const string &idf_f, const string &tfidf_f, const string &emoj_f)
  {
    onehot.save(onehot_f, arma::arma_ascii);
    t.save(t_f, arma::arma_ascii);
    tf.save(tf_f, arma::arma_ascii);
    df.save(df_f, arma::arma_ascii);
    idf.save(idf_f, arma::arma_ascii);
    tfidf.save(tfidf_f, arma::arma_ascii);
    emoj_prob.save(emoj_f, arma::arma_ascii);
  }
};

