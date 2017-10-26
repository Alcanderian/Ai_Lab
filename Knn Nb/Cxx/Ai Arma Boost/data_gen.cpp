#include <armadillo>
#include <fstream>
#include <string>
#include <cmath>
#include <exception>
#include <boost/algorithm/string.hpp>

#include "data_gen.h"

using namespace std;

void data_gen::import_csv(const string &src_csv)
{
  vector<string> s_buf, emoj_buf;
  ifstream f_src(src_csv);
  stringstream ss_buf;
  string csv_buf;
  double m_logpatterns;

  if (!f_src)
    throw exception((string("No such csv file: ") + src_csv + ".").c_str());

  // First line is table head.
  getline(f_src, csv_buf);
  boost::split(s_buf, csv_buf, boost::is_any_of(","));
  emoj_name.assign(s_buf.begin() + 1, s_buf.end());

  // Build word_idx.
  cols = rows = 0;
  pattern_set.clear();
  word_idx.clear();
  while (getline(f_src, csv_buf))
  {
    boost::replace_first(csv_buf, ",", "\t"); // "x,y,z..." -> "x\ty,z...".
    boost::replace_all(csv_buf, ",", " "); // "x\ty,z..." -> "x\ty z...".
    boost::split(s_buf, csv_buf, boost::is_any_of("\t")); // Get "x", "y z...".

    pattern_set.push_back(vector<string>());
    emoj_buf.push_back(s_buf[1]);
    boost::split(pattern_set[cols], s_buf[0], boost::is_any_of(" "));

    // For each new word, word_idx[word] = word_idx.size, then we will get their indices.
    for (int i = 0; i < pattern_set[cols].size(); ++i)
      if (word_idx.find(pattern_set[cols][i]) == word_idx.end())
        word_idx[pattern_set[cols][i]] = word_idx.size();
    ++cols;
  }
  // Note: const int &patterns = cols, &words = rows.
  rows = word_idx.size(), m_logpatterns = log(patterns);
  f_src.close();

  // Note that our vector is stored by columns, not by rows.
  onehot = arma::mat(rows, cols).zeros();
  t = arma::mat(rows, cols).zeros();
  tf = arma::mat(rows, cols).zeros();
  df = arma::vec(rows).zeros();
  idf = arma::vec(rows).zeros();
  tfidf = arma::mat(rows, cols).zeros();
  emoj_prob = arma::mat(emoj_name.size(), cols);

  // Get onthot, prepare for tf and idf.
  for (int j_word, i = 0; i < patterns; ++i)
    for (int j = 0; j < pattern_set[i].size(); ++j)
    {
      j_word = word_idx[pattern_set[i][j]];
      onehot(j_word, i) = 1;
      if (t(j_word, i)++ == 0)
        df(j_word)++;
    }

  // Get idf.
  for (int i = 0; i < words; ++i)
    idf(i) = m_logpatterns - log(df(i) + 1);

  // Get tf and tfidf.
  for (int i = 0; i < patterns; ++i)
    for (int j = 0; j < words; ++j)
      if (t(j, i) != 0)
      {
        tf(j, i) = t(j, i) / pattern_set[i].size();
        tfidf(j, i) = tf(j, i) * idf(j);
      }

  /*
  Normalize tfidf, using tfidf.col(i) = tfidf.col(i) / norm(tdidf.col(i), 2).
  By armadillo's default, for matrix X, return its normalised version,
  where each column has been normalised to have unit p-norm(by default p=2 is used).
  */
  if (data_gen_cfg.vector_type == data_gen_config::normalised_vector)
  {
    tfidf = arma::normalise(tfidf);
    onehot = arma::normalise(onehot);
    tf = arma::normalise(tf);
  }

  // Put emotion prob into matrix.
  for (int i = 0; i < patterns; ++i)
  {
    ss_buf.clear();
    ss_buf.str(emoj_buf[i]);
    for (int j = 0; j < emoj_name.size(); ++j)
      ss_buf >> emoj_prob(j, i);
  }
}
