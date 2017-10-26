#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <boost/algorithm/string.hpp>

#include "fix_classi_csv.h"

using namespace std;

void fix_classi_csv(const string &src_csv, const string &dst_csv)
{
  const char *emoj[] = {
    "anger", "disgust", "fear", "joy", "sad", "surprise"
  };
  unordered_map<string, int> emoj_idx;
  ifstream f_src(src_csv);
  ofstream f_dst(dst_csv);
  vector<string> v_buf;
  string s_buf;
  int idx;

  for (int i = 0; i < 6; ++i)
    emoj_idx[emoj[i]] = i;

  getline(f_src, s_buf);
  f_dst << "Words (split by space),anger,disgust,fear,joy,sad,surprise\n";

  while (getline(f_src, s_buf))
  {
    boost::split(v_buf, s_buf, boost::is_any_of(","));
    idx = emoj_idx[v_buf[1]];
    f_dst << v_buf[0] << ",";
    for (int i = 0; i < 6; ++i)
      f_dst << (i == idx) << (i != 5 ? "," : "\n");
  }

  f_src.close(), f_dst.close();
}
