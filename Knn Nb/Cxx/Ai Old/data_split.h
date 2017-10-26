#pragma once

#include <unordered_map>

#include "base.h"
#include "sparse_mat.hpp"

enum mod_type
{
  mod_none,
  mod_both,
  mod_classi,
  mod_regres
};

enum mat_type
{
  mat_arr_sp,
  mat_arr_only
};

struct file_cfg
{
  char *source_file = NULL;

  int buf_size = 1000;

  char *onehot_file = NULL;
  char *tf_file = NULL;
  char *tfidf_file = NULL;

  char *sp_onehot_file = NULL;
  char *sp_tf_file = NULL;
  char *sp_tfidf_file = NULL;
};

struct data_out
{
  unordered_map<string, int> word_idx;
  vector<string> classi_lables;
  double *emoji_prob;

  int m_cols, m_rows;

  ispmat sp_onehot;
  spmat sp_tf, sp_tfidf;

  int *onehot = NULL;
  double *tf = NULL, *tfidf = NULL;
};

char *after_nth_char(char *s, int n, const char &c);

void data_split(char *source_file, int buf_size,
  char *onehot_file, char *tf_file, char *tfidf_file, char *smat_file);
