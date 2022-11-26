#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include "wmath.hpp"
#include "wmath_color.hpp"
#include "dlib/matrix.h"

using dlib::abs;
using dlib::cholesky_decomposition;
using dlib::diag;
using dlib::diagm;
using dlib::dot;
using dlib::eigenvalue_decomposition;
using dlib::identity_matrix;
using dlib::inv;
using dlib::is_finite;
using dlib::length;
using dlib::length_squared;
using dlib::make_symmetric;
using dlib::matrix;
using dlib::matrix_exp;
using dlib::matrix_op;
using dlib::normalize;
using dlib::ones_matrix;
using dlib::op_make_symmetric;
using dlib::round;
using dlib::set_colm;
using dlib::squared;
using dlib::sum;
using dlib::tmp;
using dlib::trace;
using dlib::trans;
using dlib::zeros_matrix;

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
using std::isfinite;
using std::istream;
using std::lower_bound;
using std::max_element;
using std::normal_distribution;
using std::numeric_limits;
using std::ofstream;
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
using std::tuple;
using std::vector;

using wmath::mean_variance;
using wmath::signum;
using wmath::universal_distribution;
using wmath::revert_srgb_gamma;
using wmath::long_mul;

constexpr double pi = 3.141592653589793;

double make_finite(const double r){
  if (isfinite(r)) return r;
  return 0.0;
}

uint64_t saturating_add(uint64_t a,uint64_t b) {
  if (a+b<a) return ~uint64_t(0);
  return a+b;
}

uint64_t saturating_sub(uint64_t a,uint64_t b) {
  if (a>b) return a-b;
  return 0;
}

int main() {
  constexpr size_t nfs = 2400;
  constexpr size_t nss = 1600;
  matrix<double,2,1> x0  {2000.0,1200.0};
  matrix<double,2,1> p   {1667.0,690.0};
  p-=x0;
  matrix<double,2,1> kin {1130.0,1200.0};
  kin=-(kin-x0);
  matrix<double,2,1> dk  {1811.0,659.0};
  dk-=x0;
  matrix<double,2,2> S
  {
    394.0,   0.0,
      0.0, 157.0
  };
  S = S*trans(S);
  matrix<double,2,2> St
  {
      0.0,   0.0,
      0.0, 157.0
  };
  St = St*trans(St);
  matrix<double,2,1> dw = dk/length(kin);
  St+=pow(393,2u)*dw*trans(dw);
  matrix<double,2,2> Sp
  {
    126.0,  0.0,
      0.0, 45.0
  };
  double a = 0.38;
  matrix<double,2,2> R
  {
     cos(a), sin(a),
    -sin(a), cos(a)
  };
  Sp = R*Sp;
  Sp = Sp*trans(Sp);
  cerr << p << endl;
  cerr << Sp << endl;
  cerr << St << endl;
  cerr << S << endl;
  uint64_t* data = new uint64_t[nfs*nss*3];
  for (size_t i=0;i!=nfs*nss*3;++i) data[i]=uint64_t(0);
  for (size_t ss=0;ss!=nss;++ss) {
    for (size_t fs=0;fs!=nfs;++fs) {
      matrix<double,2,1> x{fs+0.5,ss+0.5};
      double v0;
      v0 = (1.0-1e-8)*pow(2.0,64)*exp(-trans(x+kin-x0)*inv( S)*(x+kin-x0));
      v0 = make_finite(v0);
      //data[3*(ss*nfs+fs)+1]=saturating_sub(data[3*(ss*nfs+fs)+1],uint64_t(v));
      //data[3*(ss*nfs+fs)+2]=saturating_sub(data[3*(ss*nfs+fs)+2],uint64_t(v));
      data[3*(ss*nfs+fs)+0]=saturating_add(data[3*(ss*nfs+fs)+0],uint64_t(v0));
      v0 = (1.0-1e-8)*pow(2.0,64)*exp(-trans(x-dk-x0)*inv(St)*(x-dk-x0));
      v0 = make_finite(v0);
      //data[3*(ss*nfs+fs)+1]=saturating_sub(data[3*(ss*nfs+fs)+1],uint64_t(v));
      //data[3*(ss*nfs+fs)+2]=saturating_sub(data[3*(ss*nfs+fs)+2],uint64_t(v));
      data[3*(ss*nfs+fs)+0]=saturating_add(data[3*(ss*nfs+fs)+0],uint64_t(v0));
      double v1;
      v1 = (1.0-1e-8)*pow(2.0,64)*exp(-trans(x-p-x0)*inv(Sp)*(x-p-x0));
      v1 = make_finite(v1);
      //data[3*(ss*nfs+fs)+0]=saturating_sub(data[3*(ss*nfs+fs)+0],uint64_t(v));
      //data[3*(ss*nfs+fs)+1]=saturating_sub(data[3*(ss*nfs+fs)+1],uint64_t(v));
      data[3*(ss*nfs+fs)+2]=saturating_add(data[3*(ss*nfs+fs)+2],uint64_t(v1));
      //data[3*(ss*nfs+fs)+1]=saturating_add(
      //    data[3*(ss*nfs+fs)+1],uint64_t((v0*v1)*pow(0.5,63)));
    }
  }
  uint8_t* rgb = new uint8_t[nfs*nss*3]();
  for (size_t i=0;i!=nfs*nss*3;++i) {
    //cerr << data[i] << " " << revert_srgb_gamma(data[i]) << endl;
    rgb[i]=revert_srgb_gamma(data[i])>>56;
  }
  cout << "P6 " << nfs << " " << nss << " " << 255 << endl;
  cout.write(reinterpret_cast<const char*>(rgb),nfs*nss*3);
}
