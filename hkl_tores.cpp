#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <dlib/optimization.h>

#include "partiality.hpp"
#include "geometry.hpp"
#include "wmath.hpp"
#include "wmath_optimisation.hpp"

using std::abs;
using std::array;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::fill;
using std::fixed;
using std::get;
using std::getline;
using std::ifstream;
using std::isinf;
using std::isnan;
using std::istream;
using std::lower_bound;
using std::make_tuple;
using std::max_element;
using std::nan;
using std::normal_distribution;
using std::numeric_limits;
using std::ofstream;
using std::random_device;
using std::round;
using std::setprecision;
using std::setw;
using std::sort;
using std::stod;
using std::streamsize;
using std::string;
using std::stringstream;
using std::swap;
using std::to_string;
using std::transform;
using std::tuple;
using std::vector;

using dlib::abs;
using dlib::cholesky_decomposition;
using dlib::identity_matrix;
using dlib::is_finite;
using dlib::length;
using dlib::length_squared;
using dlib::matrix;
using dlib::normalize;
using dlib::ones_matrix;
using dlib::squared;
using dlib::sum;
using dlib::tmp;
using dlib::trans;
using dlib::zeros_matrix;

using partiality::IDX;
using partiality::crystl;
using partiality::deserialize_crystls;
using partiality::deserialize_crystl;
using partiality::deserialize_sources;
using partiality::predict;
using partiality::predict_integrated;
using partiality::source;

using whash::patchmap;

using wmath::clip;
using wmath::count_stop_strategy;
using wmath::mean_variance;
using wmath::signum;
using wmath::pow;
using wmath::universal_distribution;

int main() {
  patchmap<IDX,tuple<double,double>> hkl_res;
  while (cin) {
    auto sources = deserialize_sources(cin);
    if (!cin) break;
    auto crystl  = deserialize_crystl(cin);
    if (!cin) break;
    double b = 0,c = 0;
    cin.read(reinterpret_cast<char*>(&b),sizeof(double));
    if (!cin) break;
    cin.read(reinterpret_cast<char*>(&c),sizeof(double));
    if (!cin) break;
    uint64_t m;
    cin.read(reinterpret_cast<char*>(&m),sizeof(uint64_t));
    int32_t h,k,l;
    float i,s;
    for (size_t j=0;j!=m;++j) {
      cin.read(reinterpret_cast<char*>(&h),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&k),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&l),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&i),sizeof(float));
      cin.read(reinterpret_cast<char*>(&s),sizeof(float));
      const matrix<double,3,1> dhkl{1.0*h,1.0*k,1.0*l};
      const matrix<double,3,1> x = crystl.R*dhkl;
      get<0>(hkl_res[{h,k,l}])+=length(x);
      get<1>(hkl_res[{h,k,l}])+=1;
    }
  }
  for (auto it=hkl_res.begin();it!=hkl_res.end();++it) {
    cout << get<0>(it->first) << " "
         << get<1>(it->first) << " "
         << get<2>(it->first) << " "
         << get<0>(it->second)/get<1>(it->second) << endl;
  }
}
