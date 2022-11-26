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
  const crystfel_geometry cgeom = get_crystfel_geometry(cin);
  cerr << "read crystfel geometry" << endl;
  if (argc<2) {
    cerr << "please specify datafile" << endl;
    return 2;
  }
  ifstream datafile(argv[1]);
  vector<uint8_t> data(1);
  while (datafile.read(reinterpret_cast<char*>(&data.back()),sizeof(uint8_t))) {
    data.resize(data.size()+1);
  }
  data.resize(data.size()-1);
  // 1480 1552
  for (auto it=cgeom.transforms.begin();it!=cgeom.transforms.end();++it) {
    for (size_t ss=it->min_ss;ss<=it->max_ss;++ss) {
      for (size_t fs=it->min_fs;fs<=it->max_fs;++fs) {
        //cerr << 1480*ss+fs << " " << 1480*1522 << endl;
        cout.write(reinterpret_cast<const char*>(
              //&data[1480*ss+fs]),
              &data[1551*ss+fs]),
              sizeof(uint8_t));
      }
    }
  }
}
