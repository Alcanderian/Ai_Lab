#include <string>
#include <cmath>
#include <vector>
#include <armadillo>
#include <unordered_set>
#include <boost/algorithm/string.hpp>

#include "nb_predict.h"

using namespace std;

void nb_predict::import_csv(const string &src_csv)
{
  int idx;

  data_gen::import_csv(src_csv);
  emoj_word_cnt = arma::vec(emoj_name.size()).zeros();
  word_emoj_cnt = arma::mat(rows, emoj_name.size()).zeros();

  for (int i = 0; i < patterns; ++i)
  {
    idx = emoj_prob.col(i).index_max();

    emoj_word_cnt(idx) += arma::sum(t.col(i)); // nw_ei.
    word_emoj_cnt.col(idx) += t.col(i); // nw_ei(x_k).
  }

  patterns_size = arma::sum(t).t();

  prob_emoj = emoj_word_cnt / arma::sum(emoj_word_cnt);
}

nb_result nb_predict::classification_one(const vector<string> &tst_pattern)
{
  nb_result ret;
  int idx, unknown_cnt = 0;
  vector<string> tst = tst_pattern;

  std::unique(tst.begin(), tst.end());
  ret.prob = arma::log(prob_emoj);
  for (int i = 0; i < tst.size(); ++i)
    if (word_idx.find(tst[i]) != word_idx.end())
    {
      idx = word_idx[tst[i]];
      ret.prob += arma::log(word_emoj_cnt.row(idx).t() + nb_cfg.laplace_alpha);
    }
    else
      unknown_cnt++;

  ret.prob += unknown_cnt * log(nb_cfg.laplace_alpha);
  ret.prob -= (tst.size() - unknown_cnt) * arma::log(emoj_word_cnt + words * nb_cfg.laplace_alpha);

  ret.prob -= ret.prob.min();
  ret.prob = arma::exp(ret.prob);

  ret.label = emoj_name[ret.prob.index_max()];

  return ret;
}

nb_result nb_predict::regression_one(const vector<string> &tst_pattern)
{
  arma::vec tst_prob;
  nb_result ret;
  int idx, unknown_cnt = 0;
  vector<string> tst = tst_pattern;

  std::unique(tst.begin(), tst.end());
  tst_prob = arma::vec(patterns).zeros();
  for (int i = 0; i < tst.size(); ++i)
    if (word_idx.find(tst[i]) != word_idx.end())
    {
      idx = word_idx[tst[i]];
      tst_prob += arma::log(t.row(idx).t() + nb_cfg.laplace_alpha);
    }
    else
      unknown_cnt++;

  tst_prob += unknown_cnt * log(nb_cfg.laplace_alpha);
  tst_prob -= (tst.size() - unknown_cnt) * arma::log(patterns_size + words * nb_cfg.laplace_alpha);

  tst_prob -= tst_prob.min();
  tst_prob = arma::exp(tst_prob);

  ret.prob = arma::vec(emoj_name.size());
  ret.prob = emoj_prob * tst_prob;

  ret.prob = arma::normalise(ret.prob, 1);

  return ret;
}

void nb_predict::classification_verify(const string &veri_csv, const string &result_csv)
{
  int pos = 0, neg = 0;
  ifstream f_veri(veri_csv);
  ofstream f_result(result_csv);
  vector<string> v_buf;
  nb_result n_buf;
  string s1_buf, s2_buf;

  getline(f_veri, s1_buf);
  f_result << "Words (split by space),actual label,classified label\n";
  while (getline(f_veri, s1_buf))
  {
    boost::split(v_buf, s1_buf, boost::is_any_of(","));
    f_result << s1_buf;

    s1_buf = v_buf[0], s2_buf = v_buf[1];
    boost::split(v_buf, s1_buf, boost::is_any_of(" "));
    n_buf = classification_one(v_buf);

    f_result << "," << n_buf.label << endl;
    if (n_buf.label == s2_buf) ++pos;
    else ++neg;
  }

  cout
    << "alpha\t" << nb_cfg.laplace_alpha << endl
    << "correct/total\t" << pos << "/" << pos + neg << endl
    << "accuracy\t" << double(pos) / (pos + neg) << endl;

  f_result
    << "alpha," << nb_cfg.laplace_alpha << endl
    << "correct/total," << pos << "/" << pos + neg << endl
    << "accuracy," << double(pos) / (pos + neg) << endl;

  f_veri.close(), f_result.close();
}

void nb_predict::classification(const string &tst_csv, const string &result_csv)
{
  ifstream f_tst(tst_csv);
  ofstream f_result(result_csv);
  vector<string> v_buf;
  nb_result n_buf;
  string s_buf;

  getline(f_tst, s_buf);
  f_result << "textid,label\n";
  while (getline(f_tst, s_buf))
  {
    boost::split(v_buf, s_buf, boost::is_any_of(","));
    f_result << v_buf[0] << ",";

    s_buf = v_buf[1];
    boost::split(v_buf, s_buf, boost::is_any_of(" "));
    n_buf = classification_one(v_buf);

    f_result << n_buf.label << endl;
  }

  f_tst.close(), f_result.close();
}

void nb_predict::regression_verify(const string &veri_csv, const string &result_csv)
{
  arma::mat veri_prob, result_prob, cor_prob;
  vector<string> veri_buf, v_buf;
  vector<arma::vec> result_buf;
  ifstream f_veri(veri_csv);
  ofstream f_result(result_csv);
  stringstream ss_buf;
  nb_result n_buf;
  string s_buf;

  getline(f_veri, s_buf);
  boost::replace_first(s_buf, ",", "\t");
  boost::split(v_buf, s_buf, boost::is_any_of("\t"));
  f_result << v_buf[1] << endl;
  while (getline(f_veri, s_buf))
  {
    boost::replace_first(s_buf, ",", "\t"); // "x,y,z..." -> "x\ty,z...".
    boost::replace_all(s_buf, ",", " "); // "x\ty,z..." -> "x\ty z...".
    boost::split(v_buf, s_buf, boost::is_any_of("\t")); // Get "x", "y z...".

    s_buf = v_buf[0];
    veri_buf.push_back(v_buf[1]);
    boost::split(v_buf, s_buf, boost::is_any_of(" "));
    n_buf = regression_one(v_buf);

    for (int i = 0; i < emoj_name.size(); ++i)
      f_result << n_buf.prob(i) << (i == int(emoj_name.size()) - 1 ? "\n" : ",");
    result_buf.push_back(n_buf.prob);
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
    << "alpha\t" << nb_cfg.laplace_alpha << endl
    << "correls\t";
  for (int i = 0; i < emoj_name.size(); ++i)
    cout << cor_prob(i, i) << (i == int(emoj_name.size()) - 1 ? "\n" : ",");
  cout << "avg_correl\t" << arma::trace(cor_prob) / emoj_name.size() << endl; // trace(A) -> sum(A.diag()).

  f_result
    << "alpha," << nb_cfg.laplace_alpha << endl
    << "correls,";
  for (int i = 0; i < emoj_name.size(); ++i)
    f_result << cor_prob(i, i) << (i == int(emoj_name.size()) - 1 ? "\n" : ",");
  f_result << "avg_correl," << arma::trace(cor_prob) / emoj_name.size() << endl; // trace(A) -> sum(A.diag()).

  f_veri.close(), f_result.close();
}

void nb_predict::regression(const string &tst_csv, const string &result_csv)
{
  ifstream f_tst(tst_csv);
  ofstream f_result(result_csv);
  vector<string> v_buf;
  nb_result n_buf;
  string s_buf;

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
    n_buf = regression_one(v_buf);

    for (int i = 0; i < emoj_name.size(); ++i)
      f_result << "," << n_buf.prob(i);
    f_result << endl;
  }

  f_tst.close(), f_result.close();
}
