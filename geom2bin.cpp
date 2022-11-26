#include "asu.hpp"
#include "encode.hpp"
#include "geometry.hpp"
#include "partiality.hpp"
#include "wmath.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::cin;
using std::cout;

int main(int argc, char *argv[])
{
  geometry::geom_ascii2bin(cin,cout);
  return 0;
}
