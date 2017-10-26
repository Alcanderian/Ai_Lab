#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "base.h"

using namespace std;

// Tri-tuple.
template<class T>
struct smat_tuple
{
  int row = 0, col = 0;
  T val;

  smat_tuple() {}
  smat_tuple(const smat_tuple &t) : row(t.row), col(t.col), val(t.val) {}
  smat_tuple(const int &r, const int &c, const T &val)
    : row(r), col(c), val(val) {}

  smat_tuple &operator=(const smat_tuple &t)
  {
    row = t.row, col = t.col, val = t.val;
    return *this;
  }
};

template<class T>
inline bool operator<(const smat_tuple<T> &a, const smat_tuple<T> &b)
{
  if (a.row == b.row)
    return a.col < b.col;
  return a.row < b.row;
}

template<class T>
class sparse_mat
{
private:
  int rows = 0, cols = 0;
  vector<smat_tuple<T>> data;

  int find(const int &r, const int &c)
  {
    int idx = 0;

    while (idx != data.size())
    {
      if (data[idx].row == r && data[idx].col == c)
        break;
      ++idx;
    }

    return idx;
  }

  void set_nth(const int &n, const T &val)
  {
    assert(n >= 0 && n < data.size());

    if (val != T(0))
      data[n].val = val;
    else
      data.erase(data.begin() + n); // Delete the node if new val == 0.

    return;
  }

public:
  sparse_mat() {}
  sparse_mat(const char *src_file) { load_file(src_file); }
  sparse_mat(const int &r, const int &c) : rows(r), cols(c) {}
  sparse_mat(T *arr, const int &r, const int &c) { load_arr1d(arr, r, c); }
  sparse_mat(const sparse_mat &m) : rows(m.rows), cols(m.cols), data(m.data) {}
  sparse_mat &unique() { unique(data.begin(), data.end()); return *this; }
  const int get_rows() { return rows; }
  const int get_cols() { return cols; }
  const size_t size() { return data.size(); }
  friend ostream &operator<<(ostream &os, sparse_mat &m) { os << m.save_str(); return os; }

  sparse_mat &operator=(const sparse_mat &m)
  {
    rows = m.rows, cols = m.cols, data = m.data;

    return *this;
  }

  sparse_mat &resize(const int &r, const int &c)
  {
    assert(r >= 0 && c >= 0);
    vector<smat_tuple<T>> new_data;

    rows = r, cols = c;
    for (int i = 0; i < data.size(); ++i) // We need to delete the elements which are out of range.
      if (data[i].row < rows && data[i].col < cols)
        new_data.push_back(data[i]);
    data = new_data;

    return *this;
  }

  sparse_mat &set(const int &r, const int &c, const T &val)
  {
    assert(r >= 0 && r < rows && c >= 0 && c < cols);
    int idx = find(r, c);

    if (idx != data.size())
    {
      if (val != T(0))
        data[idx].val = val;
      else
        data.erase(data.begin() + idx); // Delete it when the new val == 0.
    }
    else if (val != T(0)) { data.push_back(smat_tuple<T>(r, c, val)); } // Append if val != 0.

    return *this;
  }

  // Just append to data, use carefully!
  sparse_mat &append(const int &r, const int &c, const T &val)
  {
    assert(r >= 0 && r < rows && c >= 0 && c < cols);
    data.push_back(smat_tuple<T>(r, c, val));

    return *this;
  }

  const T get(const int &r, const int &c) {
    assert(r >= 0 && r < rows && c >= 0 && c < cols);
    int idx = find(r, c);

    if (idx != data.size())
      return data[idx].val;
    else return T(0);
  }

  // Load mat from matrix span in 1D.
  sparse_mat &load_arr1d(T *arr, const int &r, const int &c)
  {
    assert(r >= 0 && c >= 0);
    mat1d_index loc(c);

    rows = r, cols = c;
    for (int i = 0; i < rows; ++i)
    {
      loc.set_n_rows(i);
      for (int j = 0; j < cols; ++j)
        if (arr[loc(j)] != T(0))
          data.push_back(smat_tuple<T>(i, j, arr[loc(j)]));
    }

    return *this;
  }

  // Load from sparse mat file.
  sparse_mat &load_file(const char *src_file)
  {
    ifstream f_src(src_file);
    stringstream ss_buf;
    vector<string> v_str(3);
    string buf;
    int r, c;
    T item;

    assert(f_src);

    for (int i = 0; i < 3; ++i)
    {
      getline(f_src, v_str[i]);
      boost::trim_if(v_str[i], boost::is_any_of("[] "));
    }
    ss_buf.str(boost::join(v_str, " "));
    ss_buf >> rows >> cols >> buf;

    while (getline(f_src, buf))
    {
      boost::trim_if(buf, boost::is_any_of("[] "));
      boost::split(v_str, buf, boost::is_any_of(", "), boost::token_compress_on);
      ss_buf.clear();
      ss_buf.str(boost::join(v_str, " "));

      ss_buf >> r >> c >> item;
      if (item != T(0))
        data.push_back(smat_tuple<T>(r, c, item));
    }

    return *this;
  }

  // Convert to string.
  string &save_str(string &s = string())
  {
    char *buf = new char[100];

    sort(data.begin(), data.end());
    sprintf_s(buf, 100, "[%d]\n[%d]\n[%zd]\n", rows, cols, data.size());
    s.append(buf);

    for (int i = 0; i < data.size(); ++i)
    {
      sprintf_s(buf, 100, "[%d, %d, %s]\n", data[i].row, data[i].col, to_string(data[i].val).c_str());
      s.append(buf);
    }
    free(buf);

    return s;
  }

  sparse_mat operator+(const sparse_mat &m)
  {
    assert(rows == m.rows || cols == m.cols);
    sparse_mat ret(m);
    int idx = 0;

    for (int i = 0; i < data.size(); ++i)
    {
      idx = ret.find(data[i].row, data[i].col);
      if (idx != ret.data.size())
        ret.set_nth(idx, ret.data[idx].val + data[i].val); // Add if element exist.
      else
        ret.data.push_back(smat_tuple<T>(data[i].row, data[i].col, data[i].val)); // Rather append.
    }

    return ret;
  }
};

typedef sparse_mat<double> spmat;
typedef sparse_mat<int> ispmat;
