#include <iostream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "test_boost.h"

int test_boost()
{
  std::string s = "Hello, the beautiful world!";
  std::vector<std::string> rs;
  boost::split(rs, s, boost::is_any_of(" ,!"), boost::token_compress_on);

  std::cout << "Origin string:\n " << s << "\n\n";
  std::cout << "Split string by \" ,!\":\n";
  for (int i = 0; i < rs.size(); ++i)
    std::cout << ' ' << i << ": " << rs[i] << std::endl;
  std::cout << std::endl;

  return 0;
}
