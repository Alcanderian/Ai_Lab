#include <fstream>
#include <iomanip>
#include <cstdio>
#include <unordered_map>
#include <string>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <boost/algorithm/string.hpp>

#include "base.h"
#include "data_split.h"
#include "sparse_mat.hpp"

using namespace std;

// Return the ptr after n-th char c.
char *after_nth_char(char *s, int n, const char &c) 
{
  while (*s != 0)
    if (*s++ == c && !--n)
      break;

  return s;
}

void data_split(char *source_file, int buf_size,
  char *onehot_file, char *tf_file, char *tfidf_file, char *smat_file)
{
  high_precision_timer timer;
  unordered_map<string, int> dict; // Use hash_map to map the index of words.
  vector<vector<string>> v_word;
  ifstream f_src(source_file);
  mat1d_index loc;
  ispmat m_smat;
  FILE *f_onh, *f_tf, *f_tfidf, *f_smat;
  double *tf, *tfidf, *idf, m_logrows; // Matrix span in 1D.
  int *onehot, m_cols = 0, m_rows = 0, m_size = 0;
  char *tmp, *buf = new char[buf_size];
  char *b_onh, *b_tf, *b_tfidf; // Write buffers.
  char *t_onh, *t_tf, *t_tfidf; // Temp ptr of buffers.

  assert(f_src);
  timer.start();

  fopen_s(&f_onh, onehot_file, "w");
  fopen_s(&f_tf, tf_file, "w");
  fopen_s(&f_tfidf, tfidf_file, "w");
  fopen_s(&f_smat, smat_file, "w");

  while (f_src.getline(buf, buf_size))
  {
    tmp = after_nth_char(buf, 2, '\t'); // Get string after 2nd '\t'
    v_word.push_back(vector<string>());
    boost::split(v_word[m_rows], tmp, boost::is_any_of(" ")); // Split.
    for (int i = 0; i < v_word[m_rows].size(); ++i)
      if (dict.find(v_word[m_rows][i]) == dict.end()) // Assign if the word is new.
        dict[v_word[m_rows][i]] = dict.size(); // D[word] = D.size means the n-th word will get (n-1)th index.
    ++m_rows;
  }

  printf("Prepare data cost: %.4lf ms.\n", timer.elapse_ms());
  timer.start();

  loc.cols = m_cols = dict.size(); // Pre-calculate size and prepare location item.
  m_size = m_cols * m_rows;
  m_smat.resize(m_rows, m_cols); // Resize sparse mat.

  onehot = new int[m_size]; memset(onehot, 0, m_size * sizeof(int));
  tf = new double[m_size]; memset(tf, 0, m_size * sizeof(double));
  idf = new double[m_cols]; memset(idf, 0, m_cols * sizeof(double));
  tfidf = new double[m_size]; memset(tfidf, 0, m_size * sizeof(double));

  for (int j_col, i = 0; i < m_rows; ++i)
  {
    loc.set_n_rows(i);
    for (int j = 0; j < v_word[i].size(); ++j)
    {
      j_col = dict[v_word[i][j]]; // Map word to real col-index.
      onehot[loc(j_col)] = 1;
      if (tf[loc(j_col)]++ == 0) // New one in this line if tf[i][j_col] == 0.
      {
        m_smat.append(i, j_col, 1);
        idf[j_col] += 1;
      }
    }
  }

  m_logrows = log(m_rows);
  for (int i = 0; i < m_cols; ++i)
    idf[i] = m_logrows - log(idf[i] + 1); // Using log(a/b) = loga - logb.
  for (int i = 0; i < m_rows; ++i)
  {
    loc.set_n_rows(i);
    for (int j = 0; j < m_cols; ++j)
      if (tf[loc(j)] != 0) // Only assign when tf[i][j] != 0. This can reduce time.
      {
        tf[loc(j)] = tf[loc(j)] / v_word[i].size();
        tfidf[loc(j)] = tf[loc(j)] * idf[j];
      }
  }

  printf("Deal data cost: %.4lf ms.\n", timer.elapse_ms());
  timer.start();

  // Alloc buffer and write.
  t_onh = b_onh = new char[m_size * 5];
  t_tf = b_tf = new char[m_size * 10];
  t_tfidf = b_tfidf = new char[m_size * 20];
  for (int i = 0, t_lim = m_cols - 1; i < m_rows; ++i)
  {
    loc.set_n_rows(i);
    for (int j = 0; j < m_cols; ++j)
    {
      if (onehot[loc(j)])
      {
        // Set precision as 4.
        sprintf_s(t_onh, 4, j != t_lim ? "1 " : "1\n"); t_onh += 2;
        sprintf_s(t_tf, 9, j != t_lim ? "%.4lf " : "%.4lf\n", tf[loc(j)]); t_tf += 7;
        sprintf_s(t_tfidf, 19, j != t_lim ? "%.4lf " : "%.4lf\n", tfidf[loc(j)]); t_tfidf += strlen(t_tfidf);
      }
      else
      {
        sprintf_s(t_onh, 4, (j != t_lim ? "0 " : "0\n")); t_onh += 2;
        sprintf_s(t_tf, 9, (j != t_lim ? "0 " : "0\n")); t_tf += 2;
        sprintf_s(t_tfidf, 19, (j != t_lim ? "0 " : "0\n")); t_tfidf += strlen(t_tfidf);
      }
    }
  }

  fputs(b_onh, f_onh), fputs(b_tf, f_tf), fputs(b_tfidf, f_tfidf);
  fputs(m_smat.save_str().c_str(), f_smat);
  f_src.close(), fclose(f_onh), fclose(f_tf), fclose(f_tfidf), fclose(f_smat);

  printf("Write file cost: %.4lf ms.\n", timer.elapse_ms());
  timer.start();

  free(tf), free(tfidf), free(idf), free(onehot);
  free(buf), free(b_onh), free(b_tf), free(b_tfidf);

  printf("Free memory cost: %.4lf ms.\n", timer.elapse_ms());
  printf("Total time cost: %.4lf ms.\n", timer.tot_ms());

  return;
}
