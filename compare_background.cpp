#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <patchmap.hpp>

#include "wmath.hpp"

using std::abs;
using std::accumulate;
using std::array;
using std::bernoulli_distribution;
using std::cerr;
using std::cin;
using std::cout;
using std::copy;
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
using std::max_element;
using std::normal_distribution;
using std::numeric_limits;
using std::ofstream;
using std::poisson_distribution;
using std::pow;
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
using std::uniform_real_distribution;
using std::vector;

using whash::patchmap;

int main() {
  constexpr size_t n = 18375680/sizeof(double);
  double * background = new double[n];
  double * variance   = new double[n];
  double * data       = new double[n];
  {
    ifstream file("LCLS_2013_Mar20_r0041_223336_1eea.bin");
    file.read(reinterpret_cast<char*>(data      ),n*sizeof(double));
  }
  {
    ifstream file("background_variance.bin");
    file.read(reinterpret_cast<char*>(background),n*sizeof(double));
    file.read(reinterpret_cast<char*>(variance  ),n*sizeof(double));
  }
  //for (size_t i=0;i!=n;++i) variance[i]*=2;
  //cout.write(reinterpret_cast<char*>(background),n*sizeof(double));
  //cout.write(reinterpret_cast<char*>(variance),n*sizeof(double));
  if (true) {
    patchmap<int64_t,double> hist;
    for (size_t i=0;i!=n;++i) {
      ++hist[floor(16*(data[i]-background[i])/sqrt(variance[i]))];
    }
    for (auto it=hist.begin();it!=hist.end();++it) {
      cout << it->first/16.0 << " " << it->second << endl;
    }
  }
  delete[] background;
  delete[] variance;
  delete[] data;
}
