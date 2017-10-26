#include <cstdio>

#include "data_split.h"
#include "sparse_mat.hpp"

using namespace std;

int main(int argc, const char **argv)
{
  high_precision_timer timer;
  FILE *f_mat_c;

  timer.start();
  
  spmat a("mat_a.txt"), b("mat_b.txt"), c = a + b;
  fopen_s(&f_mat_c, "mat_c.txt", "w");
  fputs(c.save_str().c_str(), f_mat_c);
  fclose(f_mat_c);

  printf("Smat a + b cost: %.4lf ms.\n", timer.elapse_ms());

  puts(""), data_split("text.txt", 300, "onehot.txt", "tf.txt", "tfidf.txt", "smatrix.txt");

  puts(""), system("pause");

  return 0;
}
