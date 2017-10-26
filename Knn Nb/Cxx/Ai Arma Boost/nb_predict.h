#pragma once

#include <string>

#include "data_gen.h"

using namespace std;

struct nb_config
{
  double laplace_alpha = 0.3;
};

struct nb_result
{
  arma::vec prob;
  string label;
};

struct nb_predict : data_gen
{
  nb_config nb_cfg;

  arma::vec emoj_word_cnt;

  arma::mat word_emoj_cnt;

  arma::vec prob_emoj;
  arma::vec patterns_size;

  void import_csv(const string &src_csv) override;
  nb_result classification_one(const vector<string> &tst_pattern);
  nb_result regression_one(const vector<string> &tst_pattern);
  void classification_verify(const string &veri_csv, const string &result_csv);
  void classification(const string &tst_csv, const string &result_csv);
  void regression_verify(const string &veri_csv, const string &result_csv);
  void regression(const string &tst_csv, const string &result_csv);

  void export_nbmat(const string &ewc_f,
    const string &wec_f, const string &pe_f, const string &ps_f)
  {
    emoj_word_cnt.save(ewc_f, arma::arma_ascii);
    word_emoj_cnt.save(wec_f, arma::arma_ascii);
    prob_emoj.save(pe_f, arma::arma_ascii);
    patterns_size.save(ps_f, arma::arma_ascii);
  }
};

