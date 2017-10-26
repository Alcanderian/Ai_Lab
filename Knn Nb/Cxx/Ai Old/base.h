#pragma once

#include <windows.h> 

struct mat1d_index
{
  int cols = 0;
  int n_rows = 0;

  mat1d_index() {}
  mat1d_index(const int &c) : cols(c) {}
  void set_n_rows(const int &n) { n_rows = n * cols; }
  inline int operator()(const int &r, const int &c) { return r * cols + c; }
  inline int operator()(const int &c) { return n_rows + c; }
};

class high_precision_timer
{
private:
  LARGE_INTEGER freq;
  LARGE_INTEGER begin;
  LARGE_INTEGER end;
  LARGE_INTEGER tot = { 0, 0 };

public:
  high_precision_timer() { QueryPerformanceFrequency(&freq); }
  void start() { QueryPerformanceCounter(&begin); }
  void reset() { tot = { 0, 0 }; }
  double tot_ms() { return (double)tot.QuadPart * 1000 / freq.QuadPart; }

  double elapse_ms()
  {
    QueryPerformanceCounter(&end);
    tot.QuadPart += end.QuadPart - begin.QuadPart;

    return ((double)(end.QuadPart - begin.QuadPart)) * 1000 / freq.QuadPart;
  }
};
