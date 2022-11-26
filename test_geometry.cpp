#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "partiality.hpp"
#include "geometry.hpp"
#include "wmath.hpp"
#include "wmath_optimisation.hpp"

using std::abs;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::fill;
using std::fixed;
using std::get;
using std::getline;
using std::ifstream;
using std::isnan;
using std::istream;
using std::numeric_limits;
using std::ofstream;
using std::round;
using std::setprecision;
using std::setw;
using std::sort;
using std::stod;
using std::streamsize;
using std::string;
using std::stringstream;
using std::swap;
using std::tuple;
using std::vector;

using geometry::crystfel_geometry;
using geometry::get_crystfel_geometry;

int main(int argc, char** argv) {
  auto geometryfile = ifstream("geometry.geom");
  auto& geometrystream = geometryfile;
  const crystfel_geometry cgeom =
    get_crystfel_geometry(geometrystream);
}
