#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <dlib/optimization.h>

#include "asu.hpp"
#include "encode.hpp"
#include "geometry.hpp"
#include "partiality.hpp"
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
using std::stoull;

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

using SYMMETRY::decode;
using SYMMETRY::get_point_group;
using SYMMETRY::reduce_encode;

constexpr double pi          = 3.14159265358979323846;

tuple<double,tuple<double,double>> nlog_exponential(
    const double x,
    const double l
    )
{
  if (x<0) return {0,{0,0}};
  const double v = x/l + log(l);
  return {v,{1.0/l,1/l-x/pow(l,2u)}};
}

inline tuple<double,tuple<double,double,double>> gauss
(
  const double x,
  const double m,
  const double v
)
{
  const double e = pow((x-m),2)/abs(v);
        double t = exp(-0.5*e)/(sqrt(2*pi*abs(v)));
  if (isnan(t)||isinf(t)) t = 0;
  return
  {
    t,
    {
      (m-x)/abs(v)*t,
      (x-m)/abs(v)*t,
      t/(2*v)*(e-1)
    }
  };
}

inline tuple<
  double,
  matrix<double,2,1>,
  matrix<double,2,2>
  > gauss
(
  const double x,
  const double m,
  const double v
)
{
  const double d = x-m;
  const double e = pow((x-m),2)/v;
        double t = exp(-0.5*e)/(sqrt(2*pi*v));
  if (isnan(t)||isinf(t)) t = 0;
  return
  {
    t,
    {
      (x-m)/v*t,
      t/(2*v)*(e-1)
    },
    {
      pow((x-m)/v,2)*t-t/v
      ((x-m)^3*%e^-((x-m)^2/(2*v)))/(2^(3/2)*v^3*sqrt(pi*v))
 -((x-m)*%e^-((x-m)^2/(2*v)))/(sqrt(2)*v^2*sqrt(pi*v))
 -(pi*(x-m)*%e^-((x-m)^2/(2*v)))/(2^(3/2)*v*(pi*v)^(3/2)),
      ((x-m)^3*%e^-((x-m)^2/(2*v)))/(2^(3/2)*v^3*sqrt(pi*v))
 -((x-m)*%e^-((x-m)^2/(2*v)))/(sqrt(2)*v^2*sqrt(pi*v))
 -(pi*(x-m)*%e^-((x-m)^2/(2*v)))/(2^(3/2)*v*(pi*v)^(3/2)),
      ((x-m)^4*%e^-((x-m)^2/(2*v)))/(2^(5/2)*v^4*sqrt(pi*v))
 -((x-m)^2*%e^-((x-m)^2/(2*v)))/(sqrt(2)*v^3*sqrt(pi*v))
 -(pi*(x-m)^2*%e^-((x-m)^2/(2*v)))/(2^(3/2)*v^2*(pi*v)^(3/2))
 +(3*pi^2*%e^-((x-m)^2/(2*v)))/(2^(5/2)*(pi*v)^(5/2))
    }
  };
}

bool test_gauss()
{
  double x = 3.1;
  double m = 5.7;
  double v = pow(2.4,2);
  const auto [value,nabla,hessian] = gauss(x,m,v);
  const double eps = 1e-8;
  cerr << (get<0>(gauss(x,m+eps,v))
          -get<0>(gauss(x,m-eps,v)))/(2*eps) << " "
       << nabla(1) << endl;
  cerr << (get<0>(gauss(x,m,v+eps))
          -get<0>(gauss(x,m,v-eps)))/(2*eps) << " "
       << nabla(2) << endl;
  return true;
}

template<class outlier>
constexpr tuple<double,tuple<double,double,double>> llg
(
  const double x,
  const double m,
  const double v,
  const double a,
  const outlier& o
)
{
  const double p_o  = o(abs(x));
  const auto [p_g,dg]  = gauss(x,m,v);
  const double p    = a*p_o + (1-a)*p_g;
  const double llg  = -log(p);
  if (isnan(p)||llg>44||llg<0||abs(x)>1.8446744E+19) return {44,{0,0,0}};
  const double p_dm = (1-a)*get<1>(dg);
  const double p_dv = (1-a)*get<2>(dg);
  const double p_da = p_o - p_g;
  return {
    llg,
    {
      -p_dm/p,
      -p_dv/p,
      -p_da/p
    }
  };
}

template<class outlier>
tuple<double,double,double> llg_numerical(
  const double x,
  const double m,
  const double v,
  const double a,
  const outlier& o
    )
{
  const double eps = 1e-8;
  return {
  (get<0>(llg(x,m+eps,v,a,o))
  -get<0>(llg(x,m-eps,v,a,o))),
  (get<0>(llg(x,m,v+eps,a,o))
  -get<0>(llg(x,m,v-eps,a,o))),
  (get<0>(llg(x,m,v,a+eps,o))
  -get<0>(llg(x,m,v,a-eps,o)))};
}

/*
template<class outlier>
double llg_d2m
(
  const double x,
  const double m,
  const double v,
  const double a,
  const outlier& o
)
{
  const double p_o  = o(abs(x));
  const auto [p_g,dg]  = gauss(x,m,v);
  const double p    = a*p_o + (1-a)*p_g;
  const double llg  = -log(p);
  if (isnan(p)||llg>44||llg<0||abs(x)>1.8446744E+19) return {0};
  const double p_dm = (1-a)*get<1>(dg);
  const double p_dv = (1-a)*get<2>(dg);
  const double p_da = p_o - p_g;
  return p_g/pow(s,2)*(pow((x-m)/s,2)*-1);
}*/

template<long n>
void set_crystal_vector(
    const struct crystl& crystl,
    matrix<double,n,1>& x
)
{
  x( 0) = crystl.R(0,0);
  x( 1) = crystl.R(0,1);
  x( 2) = crystl.R(0,2);
  x( 3) = crystl.R(1,0);
  x( 4) = crystl.R(1,1);
  x( 5) = crystl.R(1,2);
  x( 6) = crystl.R(2,0);
  x( 7) = crystl.R(2,1);
  x( 8) = crystl.R(2,2);
  x( 9) = crystl.mosaicity;
  x(10) = crystl.peak(0,0);
  x(11) = crystl.peak(0,1);
  x(12) = crystl.peak(0,2);
  x(13) = crystl.peak(1,1);
  x(14) = crystl.peak(1,2);
  x(15) = crystl.peak(2,2);
  x(16) = crystl.strain;
  x(17) = crystl.a;
  x(18) = crystl.b;
}

template<long n>
void set_crystal_from_vector(
    const matrix<double,n,1>& x,
    struct crystl& crystl
)
{
   crystl.R(0,0)    = x( 0);
   crystl.R(0,1)    = x( 1);
   crystl.R(0,2)    = x( 2);
   crystl.R(1,0)    = x( 3);
   crystl.R(1,1)    = x( 4);
   crystl.R(1,2)    = x( 5);
   crystl.R(2,0)    = x( 6);
   crystl.R(2,1)    = x( 7);
   crystl.R(2,2)    = x( 8);
   crystl.mosaicity = x( 9);
   crystl.peak(0,0) = x(10);
   crystl.peak(0,1) = x(11);
   crystl.peak(0,2) = x(12);
   crystl.peak(1,1) = x(13);
   crystl.peak(1,2) = x(14);
   crystl.peak(2,2) = x(15);
   crystl.strain    = x(16);
   crystl.a         = x(17);
   crystl.b         = x(18);
}

tuple<matrix<double,2,1>,matrix<double,6,2>> prediction_mv(
    const double g,
    const double flx,
    const double wvn,
    const double bnd,
    const double b,
    const double c
)
{
  matrix<double,2,1> mv;
  mv(0) = flx*wvn*g;
  mv(1) = pow(g,2)*(b+c*abs(flx))*abs(flx)*(pow(wvn,2)+pow(bnd,2));
  matrix<double,6,2> J;
  J(0,0) = flx*wvn;                                                  // mu_dg
  J(1,0) = wvn*g;                                                    // mu_dflx
  J(2,0) = flx*g;                                                    // mu_dwvn
  J(3,0) = 0;                                                        // mu_dbnd
  J(4,0) = 0;                                                        // mu_db
  J(5,0) = 0;                                                        // mu_dc
  J(0,1) = 2*g*(b+c*abs(flx))*abs(flx)*(pow(wvn,2)+pow(bnd,2));      // var_dg
  J(1,1) =
    signum(flx)*(pow(g,2)*(pow(wvn,2)+pow(bnd,2))*(2*c*abs(flx)+b)); // var_dflx
  J(2,1) = 2*abs(flx)*(c*abs(flx)+b)*pow(g,2)*wvn;                   // var_dwvn
  J(3,1) = 2*bnd*abs(flx)*(c*abs(flx)+b)*pow(g,2);                   // var_dbnd
  J(4,1) = abs(flx)*pow(g,2)*(pow(wvn,2)+pow(bnd,2));                // var_db
  J(5,1) = pow(flx,2)*pow(g,2)*(pow(wvn,2)+pow(bnd,2));              // var_dc
  return {mv,J};
}

struct prediction_proposer{
  std::mt19937 gen;
  std::normal_distribution<double> d;
  const inline matrix<double,19,1> operator()(matrix<double,19,1> x) {
    //cerr << "proposing" << endl;
    const double epsilon = 5e-6;
    struct crystl crystl;
    set_crystal_from_vector(x,crystl);
    {
      double eps = epsilon*pow(det(crystl.R),1.0/3);
      for (size_t j=0;j!=3;++j) {
        for (size_t i=0;i!=3;++i) {
          crystl.R(j,i)+=eps*d(gen);
        }
      }
    }
    {
      double eps = epsilon*crystl.mosaicity+1e-8;
      crystl.mosaicity+=eps*d(gen);
    }
    {
      double eps = epsilon*pow(det(crystl.peak),1.0/3);
      for (size_t j=0;j!=3;++j) {
        for (size_t i=j;i!=3;++i) {
          crystl.peak(j,i)+=eps*d(gen);
        }
      }
    }
    {
      double eps = epsilon*abs(crystl.strain)+1e-9;
      crystl.strain+=eps*d(gen);
    }
    {
      double eps = epsilon*crystl.a;
      crystl.a+=eps*d(gen);
    }
    {
      double eps = 10*epsilon*pow(det(crystl.peak),2.0/3);
      crystl.b+=eps*d(gen);
    }
    set_crystal_vector(crystl,x);
    return x;
  }
};

struct prediction_bc_proposer{
  prediction_proposer propose;
  const inline matrix<double,21,1> operator()(matrix<double,21,1> x) {
    matrix<double,19,1> _x;
    for (size_t i=0;i!=19;++i) _x(i) = x(i);
    _x = propose(_x);
    for (size_t i=0;i!=19;++i) x(i) = _x(i);
    const double epsilon = 1e-6;
    double eps;
    eps = epsilon*abs(x(19))+epsilon;
    x(19)+=eps*propose.d(propose.gen);
    x(19) = abs(x(19));
    eps = epsilon*abs(x(20))+epsilon;
    x(20)+=eps*propose.d(propose.gen);
    x(20) = abs(x(20));
    return x; 
  }
};

/*
const inline matrix<double,19,3> get_prediction_nabla(
    const IDX& hkl,
    const vector<struct source>& sources,
    const struct crystl& crystl
    )
{
  struct crystl _crystl = crystl;
  matrix<double,19,3> J;
  const double epsilon = 1e-9;
  {
    double eps = epsilon*pow(det(crystl.R),1.0/3);
    for (size_t j=0;j!=3;++j) {
      for (size_t i=0;i!=3;++i) {
        _crystl.R(j,i) = crystl.R(j,i)+eps;
        auto tmp = predict_integrated(hkl,sources,_crystl);
        J(j*3+i,0) = get<0>(tmp);
        J(j*3+i,1) = get<1>(tmp);
        J(j*3+i,2) = get<2>(tmp);
        _crystl.R(j,i) = crystl.R(j,i)-eps;
        tmp      = predict_integrated(hkl,sources,_crystl);
        J(j*3+i,0)-= get<0>(tmp);
        J(j*3+i,1)-= get<1>(tmp);
        J(j*3+i,2)-= get<2>(tmp);
        J(j*3+i,0)/= 2*eps;
        J(j*3+i,1)/= 2*eps;
        J(j*3+i,2)/= 2*eps;
        _crystl.R(j,i) = crystl.R(j,i);
      }
    }
  }
  {
    double eps = epsilon*crystl.mosaicity;
    _crystl.mosaicity = crystl.mosaicity + eps;
    auto tmp = predict_integrated(hkl,sources,_crystl);
    J(9,0) = get<0>(tmp);
    J(9,1) = get<1>(tmp);
    J(9,2) = get<2>(tmp);
    _crystl.mosaicity = crystl.mosaicity - eps;
    tmp      = predict_integrated(hkl,sources,_crystl);
    J(9,0)-= get<0>(tmp);
    J(9,1)-= get<1>(tmp);
    J(9,2)-= get<2>(tmp);
    J(9,0)/= 2*eps;
    J(9,1)/= 2*eps;
    J(9,2)/= 2*eps;
    _crystl.mosaicity = crystl.mosaicity;
  }
  {
    double eps = epsilon*pow(det(crystl.peak),1.0/3);
    size_t k = 10;
    for (size_t j=0;j!=3;++j) {
      for (size_t i=j;i!=3;++i) {
        _crystl.peak(j,i) = crystl.peak(j,i)+eps;
        auto tmp = predict_integrated(hkl,sources,_crystl);
        J(k,0) = get<0>(tmp);
        J(k,1) = get<1>(tmp);
        J(k,2) = get<2>(tmp);
        _crystl.peak(j,i) = crystl.peak(j,i)-eps;
        tmp      = predict_integrated(hkl,sources,_crystl);
        J(k,0)-= get<0>(tmp);
        J(k,1)-= get<1>(tmp);
        J(k,2)-= get<2>(tmp);
        J(k,0)/= 2*eps;
        J(k,1)/= 2*eps;
        J(k,2)/= 2*eps;
        ++k;
        _crystl.peak(j,i) = crystl.peak(j,i);
      }
    }
  }
  {
    double eps = epsilon*crystl.strain;
    _crystl.strain = crystl.strain + eps;
    auto tmp = predict_integrated(hkl,sources,_crystl);
    J(16,0) = get<0>(tmp);
    J(16,1) = get<1>(tmp);
    J(16,2) = get<2>(tmp);
    _crystl.strain = crystl.strain - eps;
    tmp      = predict_integrated(hkl,sources,_crystl);
    J(16,0)-= get<0>(tmp);
    J(16,1)-= get<1>(tmp);
    J(16,2)-= get<2>(tmp);
    J(16,0)/= 2*eps;
    J(16,1)/= 2*eps;
    J(16,2)/= 2*eps;
    _crystl.strain = crystl.strain;
  }
  {
    double eps = epsilon*crystl.a;
    _crystl.a = crystl.a + eps;
    auto tmp = predict_integrated(hkl,sources,_crystl);
    J(17,0) = get<0>(tmp);
    J(17,1) = get<1>(tmp);
    J(17,2) = get<2>(tmp);
    _crystl.a = crystl.a - eps;
    tmp      = predict_integrated(hkl,sources,_crystl);
    J(17,0)-= get<0>(tmp);
    J(17,1)-= get<1>(tmp);
    J(17,2)-= get<2>(tmp);
    J(17,0)/= 2*eps;
    J(17,1)/= 2*eps;
    J(17,2)/= 2*eps;
    _crystl.a = crystl.a;
  }
  {
    double eps = epsilon*pow(det(crystl.peak),2.0/3);
    _crystl.b = crystl.b + eps;
    auto tmp = predict_integrated(hkl,sources,_crystl);
    J(18,0) = get<0>(tmp);
    J(18,1) = get<1>(tmp);
    J(18,2) = get<2>(tmp);
    _crystl.b = crystl.b - eps;
    tmp      = predict_integrated(hkl,sources,_crystl);
    J(18,0)-= get<0>(tmp);
    J(18,1)-= get<1>(tmp);
    J(18,2)-= get<2>(tmp);
    J(18,0)/= 2*eps;
    J(18,1)/= 2*eps;
    J(18,2)/= 2*eps;
    _crystl.b = crystl.b;
  }
  return J;
}
*/

struct initial_scaling_target {
  const double& g;
  const vector<struct source>& sources;
  const vector<tuple<IDX,double,double>>& data;
  //const inline tuple<double,matrix<double,19,1>>
  const double
  operator()(const matrix<double,19,1>& x) const {
    struct crystl crystl;
    set_crystal_from_vector(x,crystl);
    double value = 0;
    matrix<double,19,1> J = zeros_matrix<double>(19,1);
    for (auto it=data.begin();it!=data.end();++it) {
      const auto [flx,wvn,bnd] =
        predict_integrated(
            get<0>(*it),
            sources,
            crystl
            );
     // matrix<double,19,3> nabla_prediction =
     //   get_prediction_nabla(get<0>(*it),sources,crystl);
      const auto [mv,_nabla_mv] = prediction_mv(1.0,flx,wvn,bnd,0.0,0.0);
      /*matrix<double,3,2> nabla_mv;
      nabla_mv(0,0) = _nabla_mv(1,0);
      nabla_mv(0,1) = _nabla_mv(1,1);
      nabla_mv(1,0) = _nabla_mv(2,0);
      nabla_mv(1,1) = _nabla_mv(2,1);
      nabla_mv(2,0) = _nabla_mv(3,0);
      nabla_mv(2,1) = _nabla_mv(3,1);*/
      const auto [llg,_nabla_exp] =
        nlog_exponential(abs(get<1>(*it)),g+sqrt(get<2>(*it))/2
            +(isnan(mv(0))?0:mv(0)));
      {
        const auto& [h,k,l] = get<0>(*it);
        //cerr << h << " " << k << " " << l << " "
        //     << get<1>(*it) << " " << mv(0) << endl;
        //cerr << flx << " " << wvn << " " << bnd << endl;
      }
      //cerr << abs(get<1>(*it)) << " " << mv(0) << " " << llg << endl;
      value += llg;
      //cerr << value << " ";
      //matrix<double,2,1> nabla_exp;
      //nabla_exp(0) = get<1>(_nabla_exp);
      //nabla_exp(1) = 0;
      //J += nabla_prediction*nabla_mv*nabla_exp;
    }
    //cerr << setprecision(16);
    //cerr << value << endl;
    //cerr << trans(J) << endl;
    //if (isnan(value)||isinf(value)) return {1e100,zeros_matrix<double>(19,1)};
    //return {value,J};
    return value;
  }
};

/*
matrix<double,19,1> get_initial_nabla(
    const initial_scaling_target& target,
    matrix<double,19,1> x
    )
{
  matrix<double,19,1> J;
  for (size_t i=0;i!=17;++i) {
    const double epsilon = 1e-10;
    x(i)+=  epsilon;
    J(i) = get<0>(target(x));
    x(i)-=2*epsilon;
    J(i)-= get<0>(target(x));
    J(i)/=2*epsilon;
    x(i)+=  epsilon;
  }
  for (size_t i=17;i!=19;++i) {
    const double epsilon = 1e-7;
    x(i)+=  epsilon;
    J(i) = get<0>(target(x));
    x(i)-=2*epsilon;
    J(i)-= get<0>(target(x));
    J(i)/=2*epsilon;
    x(i)+=  epsilon;
  }
  return J;
}
*/

struct merge_target{
  const double& a;
  const vector<tuple<
    size_t, // crystal
    IDX,    // index
    double, // intensity
    double, // variance
    double, // flx
    double, // wvn
    double, // bnd
    double, // b (error model)
    double  // c (error model)
  >>& data;
  const tuple<double,matrix<double,1,1>>
  operator()(const matrix<double,1,1>& x) const {
    matrix<double,1,1> J{0};
    double value = 0;
    // number,intensity,variance,scalefactor
    for (const auto& [n,idx,i,v,flx,wvn,bnd,b,c] : data) {
      const auto [mv,nabla_mv] = prediction_mv(1.0,x(0)*flx,wvn,bnd,b,c);
      const auto [l,nabla] =
        llg(
          i,
          mv(0),
          mv(1)+v,
          a,
          universal_distribution
        );
      J(0)+=flx*nabla_mv(1,0)*get<0>(nabla)+nabla_mv(1,1)*get<1>(nabla);
      //cerr << get<0>(idx) << " "
      //     << get<1>(idx) << " "
      //     << get<2>(idx) << " " << l << endl;
      value += l;
    }
    if (isnan(x(0))||isinf(x(0))||isnan(value))
      return {1e300,zeros_matrix<double>(1,1)};
    return {value,J};
  }
};

struct refinement_target {
  const double& a;
  const size_t& pointgroup;
  const vector<struct source>& sources;
  const patchmap<size_t,size_t>& reduce_n;
  const vector<tuple<IDX,double,double>>& data;
  const vector<double>& intensities;
  const double
  //const tuple<double,matrix<double,21,1>>
  operator()(const matrix<double,21,1>& x) const {
    //cerr << "scaling operator begin" << endl;
    struct crystl crystl;
    set_crystal_from_vector(x,crystl);
    const double b = abs(x(19));
    const double c = abs(x(20));
    double value = 0;
    matrix<double,21,1> J = zeros_matrix<double>(21,1);
    for (const auto [idx,i,v] : data) {

      size_t reduced = reduce_encode
        (get<0>(idx),get<1>(idx),get<2>(idx),pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      const double m = intensities[reduce_n.at(reduced)];
      //if (idx_n.count(idx)==0) {
      //  cerr << "index not found" << endl;
      //  cerr << get<0>(idx) << " " << get<1>(idx) << " " << get<2>(idx) << endl;
      //  cerr << "this should not happen" << endl;
      //  continue;
      //}
      //if (idx_n.at(idx)>=intensities.size()) {
      //  cerr << "index out of range" << endl;
      //  cerr << get<0>(idx) << " " << get<1>(idx) << " " << get<2>(idx) << endl;
      //  cerr << idx_n.at(idx) << endl;
      //  cerr << "this should not happen" << endl;
      //  continue;
      //}
      const auto [flx,wvn,bnd] =
        predict_integrated(
            idx,
            sources,
            crystl
            );
      //matrix<double,19,3> nabla_prediction =
      //  get_prediction_nabla(idx,sources,crystl);
      const auto [mv,_nabla_mv] = prediction_mv(1.0,m*flx,wvn,bnd,b,c);
      //matrix<double,3,2> nabla_mv;
      //nabla_mv(0,0) = m*_nabla_mv(1,0);
      //nabla_mv(0,1) = m*_nabla_mv(1,1);
      //nabla_mv(1,0) =   _nabla_mv(2,0);
      //nabla_mv(1,1) =   _nabla_mv(2,1);
      //nabla_mv(2,0) =   _nabla_mv(3,0);
      //nabla_mv(2,1) =   _nabla_mv(3,1);
      const auto [llg_value,_llg_nabla] =
        llg(i,mv(0),v+mv(1),a,universal_distribution);
      value += llg_value;
      //matrix<double,2,1> llg_nabla;
      //llg_nabla(0) = get<0>(_llg_nabla);
      //llg_nabla(1) = get<1>(_llg_nabla);
      // 19x3 3x2 2x1 = 19x1
      //const matrix<double,19,1> dcrystl = nabla_prediction*nabla_mv*llg_nabla;
      //for (size_t i=0;i!=19;++i) J(i)+=dcrystl(i);
      //J(19)+=_nabla_mv(4,1)*llg_nabla(1);
      //J(20)+=_nabla_mv(5,1)*llg_nabla(1);
    }
    //cerr << setprecision(16) << value << endl;
    //cerr << "scaling operator end" << endl;
    return value;
    //return {value,J};
  }
};

struct scaling_target {
  const double& a;
  const size_t& pointgroup;
  const vector<struct source>& sources;
  const patchmap<size_t,size_t>& reduce_n;
  const vector<tuple<IDX,double,double>>& data;
  const vector<double>& intensities;
  const struct crystl& crystl;
  // crystl.a crystl.b error.b error.c
  const tuple<double,matrix<double,4,1>>
  operator()(const matrix<double,4,1>& x) const {
    //cerr << "scaling operator begin" << endl;
    struct crystl xcrystl = crystl;
    xcrystl.a = x(0);
    xcrystl.b = x(1);
    const double b = abs(x(2));
    const double c = abs(x(3));
    double value = 0;
    matrix<double,4,1> J = zeros_matrix<double>(4,1);
    for (const auto [idx,i,v] : data) {
      size_t reduced = reduce_encode
        (get<0>(idx),get<1>(idx),get<2>(idx),pointgroup);
      if (reduce_n.count(reduced)==0) continue;
      const double m = intensities[reduce_n.at(reduced)];
      const auto [flx,wvn,bnd] =
        predict_integrated(
            idx,
            sources,
            xcrystl
            );
      const matrix<double,3,1> dhkl
        {1.0*get<0>(idx),1.0*get<1>(idx),1.0*get<2>(idx)};
      const double dflx_da = flx/xcrystl.a;
      const double dflx_db = -0.5*length_squared(xcrystl.R*dhkl)*flx;
      const auto [mv,nabla_mv] = prediction_mv(1.0,m*flx,wvn,bnd,b,c);
      const auto [llg_value,llg_nabla] =
        llg(i,mv(0),v+mv(1),a,universal_distribution);
      //cerr << get<0>(idx) << " "
      //     << get<1>(idx) << " "
      //     << get<2>(idx) << " "
      //     << llg_value << endl;
      value += llg_value;
      J(0)+=dflx_da*m*(nabla_mv(1,0)*get<0>(llg_nabla)
                      +nabla_mv(1,1)*get<1>(llg_nabla));
      J(1)+=dflx_db*m*(nabla_mv(1,0)*get<0>(llg_nabla)
                      +nabla_mv(1,1)*get<1>(llg_nabla));
      J(2)+=nabla_mv(4,1)*get<1>(llg_nabla);
      J(3)+=nabla_mv(5,1)*get<1>(llg_nabla);
    }
    J(2)*=signum(x(2));
    J(3)*=signum(x(3));
    //cerr << setprecision(16) << value << endl;
    return {value,J};
  }
};

/*
matrix<double,21,1> get_scaling_nabla(
    const scaling_target& target,
    matrix<double,21,1> x
    )
{
  matrix<double,21,1> J;
  for (size_t i=0;i!=17;++i) {
    const double epsilon = 1e-10;
    x(i)+=  epsilon;
    J(i) = get<0>(target(x));
    x(i)-=2*epsilon;
    J(i)-= get<0>(target(x));
    J(i)/=2*epsilon;
    x(i)+=  epsilon;
  }
  for (size_t i=17;i!=21;++i) {
    const double epsilon = 1e-7;
    x(i)+=  epsilon;
    J(i) = get<0>(target(x));
    x(i)-=2*epsilon;
    J(i)-= get<0>(target(x));
    J(i)/=2*epsilon;
    x(i)+=  epsilon;
  }
  return J;
}
*/

template<class target_class>
class  target_minus_one{
  target_class& target;
  template<class arg>
  operator()(arg& x){
    return get<0>(target(x))-1;
  }
}

int main(int argc,char** argv) {
  size_t spacegroup = 1, pointgroup = 1;
  if (argc>1) pointgroup=get_point_group(spacegroup=stoull(argv[1]));
  const double a = 1.0/16;
  // this is a property of the detector
  const double g = 9.9866655;
  patchmap<size_t,size_t> reduce_n;
  vector<size_t> n_reduce;
  // ...{...{index,intensity,variance,scalefactor,variance_errormodel}...}...
  vector<vector<tuple<IDX,double,double>>> rows;
  // ...{...{n_crystal,intensity,variance}...}...
  vector<vector<tuple<
    size_t,
    IDX,
    double,
    double,
    double,
    double,
    double,
    double,
    double
  >>> cols;
  vector<double> intensities;
  vector<tuple<vector<source>,tuple<crystl,double,double>>> parameters;
  //const auto subset = [](const size_t& i){return (i%2)==0;};
  //const auto subset = [](const size_t& i){return (i%2)==1;};
  const auto subset = [](const size_t& i){return true;};
  for (size_t counter=0;cin;++counter) {
    auto sources = deserialize_sources(cin);
    if (!cin) break;
    auto crystl  = deserialize_crystl(cin);
    if (!cin) break;
    double b = pow(g,-2),c = pow(g,-2);
    cin.read(reinterpret_cast<char*>(&b),sizeof(double));
    if (!cin) break;
    cin.read(reinterpret_cast<char*>(&c),sizeof(double));
    if (!cin) break;
    //crystl.a*=1e3;
    //crystl.U = trans(crystl.U);
    //crystl.R = trans(crystl.R);
    //cerr << crystl.R;
    //cerr << crystl.peak;
    //cerr << crystl.mosaicity << " " << crystl.strain << " " << crystl.a << " "
    //     << crystl.b << endl;
    if (subset(counter))
      parameters.emplace_back(sources,make_tuple(crystl,b,c));
    uint64_t m;
    cin.read(reinterpret_cast<char*>(&m),sizeof(uint64_t));
    //cerr << setw(8) << parameters.size() << setw(8) << m << endl;
    if (subset(counter)) {
      rows.push_back(vector<tuple<IDX,double,double>>{});
      rows.back().reserve(m);
    }
    int32_t h,k,l;
    float i,s;
    for (size_t j=0;j!=m;++j) {
      cin.read(reinterpret_cast<char*>(&h),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&k),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&l),sizeof(int32_t));
      cin.read(reinterpret_cast<char*>(&i),sizeof(float));
      cin.read(reinterpret_cast<char*>(&s),sizeof(float));
      //cerr << h << " " << k << " " << l << " " << i << " " << s << endl;
      const IDX idx{h,k,l};
      const matrix<double,3,1> dhkl{1.0*h,1.0*k,1.0*l};
      const matrix<double,3,1> x = crystl.R*dhkl;
      //cerr << h << " " << k << " " << l << " " << i << " " << s << " "
      //     << length(x) << endl;
      if (length(x)>4) continue;
      //if (length(x)>2) continue;
      if (subset(counter)) {
        rows.back().emplace_back(idx,i,pow(s,2));
        const size_t reduced = reduce_encode(h,k,l,pointgroup);
        if (reduce_n.count(reduced)==0) {
          reduce_n[reduced] = n_reduce.size();
          n_reduce.push_back(reduced);
          cols.emplace_back(
              vector<tuple<
                size_t,IDX,double,double,double,double,double,double,double
                >>{});
        }
        cols[reduce_n[reduced]].emplace_back(
          rows.size()-1,idx,i,pow(s,2),0.0,0.0,0.0,0.0,0.0
          );
      }
    }
    if (subset(counter)) rows.back().shrink_to_fit();
    if (rows.size()==500) break;
  }
  parameters.shrink_to_fit();
  n_reduce.shrink_to_fit();
  rows.shrink_to_fit();
  cols.shrink_to_fit();
  for (auto& row : rows) row.shrink_to_fit();
  if constexpr (false) {
  cerr << "begin initial scaling" << endl;
  transform(
      std::execution::par_unseq,
      parameters.begin(),parameters.end(),
      rows.begin(),
      parameters.begin(),
      [&g]
      (
        const tuple<vector<source>,tuple<crystl,double,double>>& parameters,
        const vector<tuple<IDX,double,double>>& data
      )
      {
        const auto& [sources,crystl_a_b] = parameters;
        auto [crystl,b,c] = crystl_a_b;
        matrix<double,19,1> x;
        set_crystal_vector(crystl,x);
        const initial_scaling_target target{g,sources,data};
        //const matrix<double,19,1> J0 = get<1>(target(x));
        //const matrix<double,19,1> J1 = get_initial_nabla(target,x);
        //cerr << trans(J0) << endl;
        //cerr << trans(J1) << endl;
        int best_i = 0;
        //double best_value = get<0>(target(x));
        double best_value = target(x);
        for (int i=-4;i<=4;++i) {
          matrix<double,19,1> _x = x;
          _x(17) = x(17)*pow(2,i);
          //if (get<0>(target(_x))<best_value) {
          if (target(_x)<best_value) {
            //best_value = get<0>(target(_x));
            best_value = target(_x);
            best_i = i;
          }
        }
        x(17)*=pow(2,best_i);
        prediction_proposer propose;
        find_min_numerical
          (
            x,
            target,
            propose,
            count_stop_strategy{pow(2ul,16u),pow(2ul,16u)}
          );
        cerr << "." ;
        //cerr << trans(x);
        set_crystal_from_vector(x,crystl);
        return make_tuple(sources,make_tuple(crystl,0,0));
      });
  }
  if constexpr (false) {
  for (const auto& [sources,crystl_a_b] : parameters) {
    const auto& [crystl,b,c] = crystl_a_b;
    cout << crystl.R
         <<crystl.peak(0,0)<<" "<<crystl.peak(0,1)<<" "<<crystl.peak(0,2)<<endl
                                <<crystl.peak(1,1)<<" "<<crystl.peak(1,2)<<endl
                                                       <<crystl.peak(2,2)<<endl;
    cout << crystl.mosaicity << " " << crystl.strain << " "
         << crystl.a         << " " << crystl.b      << " "
         << b                << " " << c             << endl;
  }
  return 0;
  }
  intensities.resize(cols.size());
  for (auto it=intensities.begin();it!=intensities.end();++it) *it=1.0;
  double targetsum;
  for (size_t i=0;i!=256;++i) {
    cerr << "update scaling factor and variance from error model in columns"
         << endl;
    ofstream premerge("premerge");
    for (size_t n=0;n!=cols.size();++n) {
      const size_t reduced = n_reduce[n];
      const auto [h,k,l] = decode(reduced,pointgroup);
      const double& m = intensities[n];
      premerge << reduced << " " << h << " " << k << " " << l << endl;
      for (size_t i=0;i!=cols[n].size();++i) {
        const auto& idx = get<1>(cols[n][i]);
        const auto& [sources,crystl_b_c] = parameters[get<0>(cols[n][i])];
        const auto& [crystl,b,c] = crystl_b_c;
        auto [flx,wvn,bnd] = predict_integrated(idx,sources,crystl);
        const matrix<double,2,1> mv = get<0>(prediction_mv(
              1.0,m*flx,wvn,bnd,abs(b),abs(c)));
        get<4>(cols[n][i]) = flx;
        get<5>(cols[n][i]) = wvn;
        get<6>(cols[n][i]) = bnd;
        get<7>(cols[n][i]) = abs(b);
        get<8>(cols[n][i]) = abs(c);
        premerge << get<0>(cols[n][i]) << " "
                 << get<0>(idx) << " "
                 << get<1>(idx) << " "
                 << get<2>(idx) << " "
                 << get<2>(cols[n][i]) << " "
                 << get<3>(cols[n][i]) << " "
                 << mv(0) << " "
                 << mv(1) << " "
                 << get<0>(gauss(
                       get<2>(cols[n][i]),
                       mv(0),
                       mv(1)+get<3>(cols[n][i])
                       )) << " "
                 << universal_distribution(abs(get<2>(cols[n][i]))) << endl;
      }
    }
    targetsum = 0.0;
    constexpr bool skipmerge = false;
    if constexpr (skipmerge) {
      for (auto it=intensities.begin();it!=intensities.end();++it) *it=1.0;
    } else {
      cerr << "merge" << endl;
      transform(
          std::execution::par_unseq,
          cols.begin(),cols.end(),
          intensities.begin(),
          intensities.begin(),
          [&a](
            const vector<tuple<
                size_t,IDX,double,double,double,double,double,double,double
                >>& data,
             const double& m) {
            matrix<double,1,1> x{m};
            merge_target target{a,data};
            //if constexpr (true){
            //  target(x);
            //  return x(0);
            //}
            //matrix<double,1,1> xp{m+1e-8};
            //matrix<double,1,1> xm{m-1e-8};
            //cerr << (get<0>(target(xp))-get<0>(target(xm)))/(2e-8) << " " 
            //     << get<1>(target(xp))(0) << endl;
            int best_i = 0;
            double best_value = get<0>(target(x));
            for (int i=-256;i<=256;++i) {
              matrix<double,1,1> _x = x;
              _x(0) = x(0)*pow(pow(2,1.0/16),i);
              if (get<0>(target(_x))<best_value) {
                best_value = get<0>(target(_x));
                best_i = i;
              }
            }
            x(0)*=pow(pow(2,1.0/16),best_i);
            find_min
              (
                x,
                target,
                count_stop_strategy{1024,1024},
                1e-9
              );
            return x(0);
          });
      ofstream intensity_file("intensities_"+to_string(i));
      for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
        const size_t reduced = it->first;
        const size_t n       = it->second;
        merge_target target{a,cols[n]};
        matrix<double,1,1> xp {intensities[n]};
        const double cv = get<0>(target(x));
        const double cd = get<1>(target(x))(0);
        x(0)+=1e-3;
        const double pd = get<1>(target(x))(0);
        x(0)-=2e-3;
        const double md = get<1>(target(x))(0);
        const double d2 = (pd-md)/2e-3;
        intensity_file << reduced << " "
                       << intensities[n] << "  " << sqrt(1/d2) << endl;
      }
      targetsum = 0.0;
      for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
        const size_t reduced = it->first;
        const size_t n       = it->second;
        merge_target target{a,cols[n]};
        matrix<double,1,1> x{intensities[n]};
        targetsum+=get<0>(target(x));
      }
      cerr << "total target = " << targetsum << endl;
    }
    constexpr bool skipscale = false;
    if constexpr (!skipscale) {
    cerr << "scale" << endl;
    transform(
        std::execution::par_unseq,
        parameters.begin(),parameters.end(),
        rows.begin(),
        parameters.begin(),
        [&a,&pointgroup,&reduce_n,&intensities]
        (
          const tuple<vector<struct source>,
                      tuple<struct crystl,double,double>
                >& parameters,
          const vector<tuple<IDX,double,double>>& data
        )
        {
          const auto& [sources,crystl_b_c] = parameters;
          auto [crystl,b,c] = crystl_b_c;
          matrix<double,4,1> x{crystl.a,crystl.b,b,c};
          scaling_target target
            {a,pointgroup,sources,reduce_n,data,intensities,crystl};
          //if constexpr (true){
          //  target(x);
          //  return make_tuple(sources,make_tuple(crystl,b,c));
          //}
          if constexpr (false) {
            for (size_t i=0;i!=4;++i) {
              matrix<double,4,1> xp = x;
              xp(i)*=1.1;
              matrix<double,4,1> xm = x;
              xm(i)*=0.9;
              cerr << get<1>(target(x))(i) << " "
                   << (get<0>(target(xp))-get<0>(target(xm)))/(xp(i)-xm(i))
                   << endl;
            }
            cerr << endl;
          }
          //cerr << "before optimisation : " << get<0>(target(x)) << endl;
          find_min
            (
              x,
              target,
              count_stop_strategy{pow(2ul,8ul),pow(2u,8ul)},
              1e-12
            );
          //cerr << "after optimisation : " << get<0>(target(x)) << endl;
          crystl.a = x(0);
          crystl.b = x(1);
          b = abs(x(2));
          c = abs(x(3));
          cerr << ".";
          return make_tuple(sources,make_tuple(crystl,b,c));
        });
    cerr << endl;
    }
    for (size_t n=0;n!=cols.size();++n) {
      const size_t reduced = n_reduce[n];
      const auto [h,k,l] = decode(reduced,pointgroup);
      const double& m = intensities[n];
      for (size_t i=0;i!=cols[n].size();++i) {
        const auto& idx = get<1>(cols[n][i]);
        const auto& [sources,crystl_b_c] = parameters[get<0>(cols[n][i])];
        const auto& [crystl,b,c] = crystl_b_c;
        auto [flx,wvn,bnd] = predict_integrated(idx,sources,crystl);
        const matrix<double,2,1> mv = get<0>(prediction_mv(
              1.0,m*flx,wvn,bnd,abs(b),abs(c)));
        get<4>(cols[n][i]) = flx;
        get<5>(cols[n][i]) = wvn;
        get<6>(cols[n][i]) = bnd;
        get<7>(cols[n][i]) = abs(b);
        get<8>(cols[n][i]) = abs(c);
      }
    }
    targetsum = 0.0;
    for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
      const size_t reduced = it->first;
      const size_t n       = it->second;
      merge_target target{a,cols[n]};
      matrix<double,1,1> x{intensities[n]};
      targetsum+=get<0>(target(x));
    }
    cerr << "total target = " << targetsum << endl;
    //return 0;
    cerr << "refine" << endl;
    transform(
        std::execution::par_unseq,
        parameters.begin(),parameters.end(),
        rows.begin(),
        parameters.begin(),
        [&a,&pointgroup,&reduce_n,&intensities]
        (
          const tuple<vector<struct source>,
                      tuple<struct crystl,double,double>
                >& parameters,
          const vector<tuple<IDX,double,double>>& data
        )
        {
          const auto& [sources,crystl_b_c] = parameters;
          auto [crystl,b,c] = crystl_b_c;
          matrix<double,21,1> x;
          set_crystal_vector(crystl,x);
          x(19) = abs(b);
          x(20) = abs(c);
          /*
          scaling_target target{a,sources,idx_n,data,intensities};
          matrix<double,21,1> J = get_scaling_nabla(target,x);
          for (size_t i=0;i!=21;++i) {
            cerr << get<1>(target(x))(i) << " " << J(i) << endl;
          }*/
          prediction_bc_proposer propose;
          find_min_numerical
            (
              x,
              refinement_target{a,pointgroup,sources,reduce_n,data,intensities},
              propose,
              count_stop_strategy{pow(2ul,8ul),pow(2u,8ul)}
            );
          set_crystal_from_vector(x,crystl);
          b = abs(x(19));
          c = abs(x(20));
          cerr << ".";
          return make_tuple(sources,make_tuple(crystl,b,c));
        });
    cerr << endl; 
    for (size_t n=0;n!=cols.size();++n) {
      const size_t reduced = n_reduce[n];
      const auto [h,k,l] = decode(reduced,pointgroup);
      const double& m = intensities[n];
      for (size_t i=0;i!=cols[n].size();++i) {
        const auto& idx = get<1>(cols[n][i]);
        const auto& [sources,crystl_b_c] = parameters[get<0>(cols[n][i])];
        const auto& [crystl,b,c] = crystl_b_c;
        auto [flx,wvn,bnd] = predict_integrated(idx,sources,crystl);
        const matrix<double,2,1> mv = get<0>(prediction_mv(
              1.0,m*flx,wvn,bnd,abs(b),abs(c)));
        get<4>(cols[n][i]) = flx;
        get<5>(cols[n][i]) = wvn;
        get<6>(cols[n][i]) = bnd;
        get<7>(cols[n][i]) = abs(b);
        get<8>(cols[n][i]) = abs(c);
      }
    }
    targetsum = 0.0;
    for (auto it=reduce_n.begin();it!=reduce_n.end();++it) {
      const size_t reduced = it->first;
      const size_t n       = it->second;
      merge_target target{a,cols[n]};
      matrix<double,1,1> x{intensities[n]};
      targetsum+=get<0>(target(x));
    }
    cerr << "total target = " << targetsum << endl;
    ofstream crystls_file("crystls_"+to_string(i));
    crystls_file << setprecision(10);
    for (const auto& [sources,crystl_b_c] : parameters) {
      const auto& [crystl,b,c] = crystl_b_c;
      crystls_file
        << "< "
        <<crystl.R
        <<crystl.peak(0,0)<<" "<<crystl.peak(0,1)<<" "<<crystl.peak(0,2)<<endl
                               <<crystl.peak(1,1)<<" "<<crystl.peak(1,2)<<endl
                                                      <<crystl.peak(2,2)<<endl
        << crystl.mosaicity << " " << crystl.strain << " "
        << crystl.a         << " " << crystl.b      << " "
        << abs(b)           << " " << abs(c)        << endl;
    }
  }
}
