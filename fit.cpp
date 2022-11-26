#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
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

using dlib::abs;
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

using whash::patchmap;
using partiality::IDX;
using partiality::crystl;
using partiality::deserialize_crystls;
using partiality::deserialize_sources;
using partiality::predict;
using partiality::source;
using wmath::universal_distribution;
using wmath::mean_variance;
using wmath::count_stop_strategy;
using wmath::clip;
using wmath::signum;

constexpr double pi = 3.141592653589793;

/*
template<class functor>
const double
//const tuple<double, // llg
//            double, // d_x
//            double, // d_g
//            vector<tuple<double,double,double>>, // d_flx, d_wvn, d_bnd
//            double, // d_bg
//            double, // d_sigma_bg
//            double, // d_a
//            double, // d_b
//            double  // d_c
//            >
llg_pixel(
    const double& x,
    const double& g,
    const vector<tuple<double,double,double>>& prediction,
    const double& flx,
    const double& wvn,
    const double& bnd,
    const double& bg,
    const double& sigma_bg,
    const double& a, // outlier probability
    const double& b, // error scaling with square root of flx
    const double& c, // error scaling with flx
    const functor& universal // universal distribution for outliers
    ) {
  double mu = bg;
  double var = pow(sigma_bg,2);
  for (auto it=prediction.begin();it!=prediction.end();++it) {
    const auto& [flx,wvn,bnd] = *it;
    mu  += flx*wvn*g;
    var += pow(g,2)*(b+c*flx)*flx*(pow(wvn,2)+pow(bnd,2));
  }
  const double uni  = universal(x);
  const double gau  = exp(-0.5*pow(x-mu,2)/var)/sqrt(2*pi*var);
  const double agau = (1-a)*gau;
  const double auni =    a *uni;
  const double p    = agau+auni;
  return -log(p);
}
*/

template<class functor>
const tuple<double, // llg
            double, // x
            double, // d_g
            double, // d_flx
            double, // d_wvn
            double, // d_bnd
            double, // d_bg
            double, // d_sigma_bg
            double, // d_a
            double, // d_b
            double  // d_c
            >
llg_pixel(
    const double& x,
    const double& g,
    const double& flx,
    const double& wvn,
    const double& bnd,
    const double& bg,
    const double& sigma_bg,
    const double& a, // outlier probability
    const double& b, // error scaling with square root of flx
    const double& c, // error scaling with flx
    const functor& universal // universal distribution for outliers
    ){
  const double mu   = bg+flx*wvn*g;
  const double var  =  pow(sigma_bg,2)
                     + pow(g,2)*(b+c*flx)*flx*(pow(wvn,2)+pow(bnd,2));
  const double uni  = universal(x);
  const double gau  = exp(-0.5*pow(x-mu,2)/var)/sqrt(2*pi*var);
  const double agau = (1-a)*gau;
  const double auni =    a *uni;
  const double p    = agau+auni;
  //cerr << sigma_bg << " " << g << " " << b << " " << c << " " << flx << " "
  //     << wvn << " " << bnd << endl;
  //cerr << x << " " << mu << " " << var << endl;
  //cerr << agau << " " << auni << endl;
  //cerr << p << endl;
  if (isnan(agau)||agau<=0) return {
     (isnan(-log(auni))||isinf(-log(auni)))?1e8:-log(auni), // llg
     0.0,       // x
     0.0,       // d_g
     0.0,       // d_flx
     0.0,       // d_wvn
     0.0,       // d_bnd
     0.0,       // d_bg
     0.0,       // d_sigma_bg
    -1/a,       // d_a
     0.0,       // d_b
     0.0        // d_c
  };
  return {
    // llg
    -log(p),
    // x
     ((x-flx*g*wvn-bg)/var*agau)/p,
    // d_g
    -(
        ((1-a)*((flx*(c*flx+b)*g*(pow(wvn,2)+pow(bnd,2))*pow(x-flx*g*wvn-bg,2))
             /pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2)
             +(flx*wvn*(x-flx*g*wvn-bg))
              /(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2)))
            *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                        +pow(sigma_bg,2)))))
 /(sqrt(2)*sqrt(pi)*sqrt(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2)))
 -((1-a)*flx*(c*flx+b)*g*(pow(wvn,2)+pow(bnd,2))
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_flx
    -(
        ((1-a)*((((c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+c*flx*pow(g,2)*(pow(wvn,2)+pow(bnd,2)))
             *pow(x-flx*g*wvn-bg,2))
             /(2*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2))
             +(g*wvn*(x-flx*g*wvn-bg))/(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))
            *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                        +pow(sigma_bg,2)))))
 /(sqrt(2)*sqrt(pi)*sqrt(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2)))
 -((1-a)*((c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+c*flx*pow(g,2)*(pow(wvn,2)+pow(bnd,2)))
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(pow(2,1.5)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_wvn
    -(
        ((1-a)*((flx*(c*flx+b)*pow(g,2)*wvn*pow(x-flx*g*wvn-bg,2))
              /pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2)
              +(flx*g*(x-flx*g*wvn-bg))/(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                        +pow(sigma_bg,2)))
             *exp(-pow(x-flx*g*wvn-bg,2)/(2
                                      *(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))))
 /(sqrt(2)*sqrt(pi)*sqrt(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2)))
 -((1-a)*flx*(c*flx+b)*pow(g,2)*wvn
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_bnd
    -(
        ((1-a)*bnd*flx*(c*flx+b)*pow(g,2)*pow(x-flx*g*wvn-bg,2)
             *exp(-pow(x-flx*g*wvn-bg,2)/(2
                                      *(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))))
 /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2.5))
 -((1-a)*bnd*flx*(c*flx+b)*pow(g,2)
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_bg
    -(
        ((1-a)*(x-flx*g*wvn-bg)
             *exp(-pow(x-flx*g*wvn-bg,2)/(2
                                      *(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))))
 /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_sigma_bg
    -(
        ((1-a)*sigma_bg*pow(x-flx*g*wvn-bg,2)
             *exp(-pow(x-flx*g*wvn-bg,2)/(2
                                      *(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))))
 /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2.5))
 -((1-a)*sigma_bg
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(sqrt(2)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_a
    -(uni-gau)/p,
    // d_b
    -(
        ((1-a)*flx*pow(g,2)*(pow(wvn,2)+pow(bnd,2))*pow(x-flx*g*wvn-bg,2)
             *exp(-pow(x-flx*g*wvn-bg,2)/(2
                                      *(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))))
 /(pow(2,1.5)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2.5))
 -((1-a)*flx*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(pow(2,1.5)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p,
    // d_c
    -(
        ((1-a)*pow(flx,2)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))*pow(x-flx*g*wvn-bg,2)
             *exp(-pow(x-flx*g*wvn-bg,2)/(2
                                      *(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                       +pow(sigma_bg,2)))))
 /(pow(2,1.5)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),2.5))
 -((1-a)*pow(flx,2)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
        *exp(-pow(x-flx*g*wvn-bg,2)/(2*(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))
                                    +pow(sigma_bg,2)))))
  /(pow(2,1.5)*sqrt(pi)*pow(flx*(c*flx+b)*pow(g,2)*(pow(wvn,2)+pow(bnd,2))+pow(sigma_bg,2),1.5))
     )/p
  };
}

void test_pixel_llg() {
  const double x        =   170;
  const double g        =   8.3;
  const double flx      =  11.1;
  const double wvn      =  10.1; // keV
  const double bnd      =   0.1;
  const double bg       = 100.0;
  const double sigma_bg =  29.0;
  const double a        =1.0/16;
  const double b        =   1.1;
  const double c        =   0.1;  
  const double epsilon  =  1e-6;
  const double p        =  1+epsilon;
  const double m        =  1-epsilon;
  const auto llg_nabla = llg_pixel(
      x,
      g,
      flx,
      wvn,
      bnd,
      bg,
      sigma_bg,
      a,
      b,
      c,
      universal_distribution
      );
  cout << get<0>(llg_nabla) << endl;
  cout << get<1>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x*p,g,flx,wvn,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x*m,g,flx,wvn,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         ))/(x*p-x*m) << endl;
  cout << get<2>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g*p,flx,wvn,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g*m,flx,wvn,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         ))/(g*p-g*m) << endl;
  cout << get<3>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx*p,wvn,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx*m,wvn,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         ))/(flx*p-flx*m) << endl;
  cout << get<4>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn*p,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn*m,bnd,bg,sigma_bg,a,b,c,universal_distribution)
         ))/(wvn*p-wvn*m) << endl;
  cout << get<5>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn,bnd*p,bg,sigma_bg,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn,bnd*m,bg,sigma_bg,a,b,c,universal_distribution)
         ))/(bnd*p-bnd*m) << endl;
  cout << get<6>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg*p,sigma_bg,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg*m,sigma_bg,a,b,c,universal_distribution)
         ))/(bg*p-bg*m) << endl;
  cout << get<7>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg*p,a,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg*m,a,b,c,universal_distribution)
         ))/(sigma_bg*p-sigma_bg*m) << endl;
  cout << get<8>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg,a*p,b,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg,a*m,b,c,universal_distribution)
         ))/(a*p-a*m) << endl;
  cout << get<9>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg,a,b*p,c,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg,a,b*m,c,universal_distribution)
         ))/(b*p-b*m) << endl;
  cout << get<10>(llg_nabla) << " "
       << (get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg,a,b,c*p,universal_distribution)
         )-get<0>(
           llg_pixel(x,g,flx,wvn,bnd,bg,sigma_bg,a,b,c*m,universal_distribution)
         ))/(c*p-c*m) << endl;
}

double common_target(
    const double a,
    const double* data,
    const patchmap<size_t,tuple<double,double>> prediction
    ) {
  double value = 0;
  for (auto it=prediction.begin();it!=prediction.end();++it) {
    const auto [mu,var] = it->second;
    const double x = data[it->first];
    double v = -log((1-a)*exp(-0.5*pow(x-mu,2)/var)
                      +a *universal_distribution(abs(x)));
    if (isnan(v)||isinf(v))
      cerr << x << " " << mu << " " << var << endl;
    if (isnan(v)||isinf(v)) v = -log(a*universal_distribution(abs(x)));
    if (isnan(v)||isinf(v)) v = 1e8;
    value += v;
  }
  return value;
}

void set_crystal_vector(
    matrix<double,21,1>& x,
    const struct crystl& crystl,
    const double& b,
    const double& c
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
  x(19) = b;
  x(20) = c;
}

void set_crystal_from_vector(
    const matrix<double,21,1>& x,
    struct crystl& crystl,
    double& b,
    double& c
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
   crystl.peak(0,1) = crystl.peak(1,0) = x(11);
   crystl.peak(0,2) = crystl.peak(2,0) = x(12);
   crystl.peak(1,1) = x(13);
   crystl.peak(1,2) = crystl.peak(2,1) = x(14);
   crystl.peak(2,2) = x(15);
   crystl.strain    = x(16);
   crystl.a         = x(17);
   crystl.b         = x(18);
   b                = x(19);
   c                = x(20);
}

/*struct background_target{
  const double& a; // outlier fraction
  const double& d; // data
  const double& f; // prediction
  const double& s; // prediction sigma
  vector<tuple<double,double>> neighbours;
  const tuple<double,matrix<double,2,1>>
  operator()(const matrix<double,2,1>& x) const {
    const double var = pow(s,2)+pow(x(1),2);
    const double g = (1-a)*exp(-0.5*pow(x(0)+(d-f),2)/var)
                                           /sqrt(2*pi*var);
    const double p = g+a*universal_distribution(d);
    double value = -log(p);
    matrix<double,2,1> J{
      (x-(d-f))/var*g/p,
      ((x(1)/var*(1-a))-x(1)*pow((x-(d-f))/var,2))*g/p
    };
    for (auto it=neighbours.begin();it!=neighbours.end();++it) {
      const double dist = 16;
      const double var = pow(x(1),2)+pow(get<1>(*it),2);
      value+=dist*0.5*(pow(x(0)-get<0>(*it),2)/var+log(2*pi*var));
      J(0)+=dist*(x(0)-get<0>(*it))/var;
      J(1)-=dist*x(1)*(pow((x(0)-get<0>(*it))/var,2)+1/var);
    }
    return {value,J};
  }
};*/

struct individual_background_target{
  const double& a; // outlier fraction
  const double d; // data
  const double pred;
  const double pred_var;
  vector<tuple<double,double>> neighbours;
  const tuple<double,matrix<double,2,1>>
  operator()(const matrix<double,2,1>& x) const {
    matrix<double,2,1> J = zeros_matrix<double>(2,1);
    const double bg       = x(0);
    const double bg_var   = abs(x(1));
    const double var      = bg_var+pred_var;
    const double gau      = (1-a)
                          * exp(-0.5*pow(bg+pred-d,2)/var)/sqrt(2*pi*var);
    const double prob     = gau+a*universal_distribution(d);
          double v        = -log(prob);
    if (isnan(v)||isinf(v)) v = -log(a*universal_distribution(d));
    if (isnan(v)||isinf(v)) v = 1e8;
    double value = v;
    J(0) += (pred-d+bg)/var*gau/prob;
    J(1) += signum(x(1))*sqrt(abs(bg_var))
           *gau*(1/var-pow((pred-d+bg)/var,2))/prob;
    for (auto [_bg,_bg_var]: neighbours) {
      //cerr << bg           << " " << _bg           << " "
      //     << sqrt(bg_var) << " " << sqrt(_bg_var) << endl;
      const double dist = 16;
      const double var = (bg_var+_bg_var)/dist;
      double v = 0.5*(pow(bg-_bg,2)/var+log(2*pi*var));
      if (isnan(v)||isinf(v)) {
        value += 1e8;
      } else {
        value += v;
        J(0)  += (bg-_bg)/var;
        J(1)  += dist*signum(x(1))*abs(bg_var)*(1/var-pow((bg-_bg)/var,2));
      }
    }
    return {value,J};
  }
};

struct background_target{
  const double& a; // outlier fraction
  const double* data; // data
  const patchmap<size_t,tuple<double,double>>& rendering;
  patchmap<size_t,tuple<double,double,vector<size_t>>>& background;
  patchmap<size_t,size_t> reverse_index; // pixel -> index
  const tuple<double,matrix<double,0,1>>
  operator()(const matrix<double,0,1>& x) const {
    double value = 0;
    double debug = 0;
    matrix<double,0,1> J = zeros_matrix<double>(x.nr(),x.nc());
    for (auto it0=rendering.begin();it0!=rendering.end();++it0) {
      const size_t i        = it0->first;
      if (background.count(i)==0) continue;
      const size_t j        = reverse_index[i];
      if (2*j+1>=x.nr()) {
        cerr << "j out ouf range, this should not happen" << endl;
        continue;
      }
      const double bg       =     x(2*j+0) ;
      const double bg_var   = abs(x(2*j+1));
      const double d        = data[i];
      const auto& [pred,pred_var] = it0->second;
      const double var      = bg_var+pred_var;
      const double gau      =
        (1-a)*exp(-0.5*pow(bg+pred-d,2)/var)/sqrt(2*pi*var);
      const double prob     = gau+a*universal_distribution(d);
            double v        = -log(prob);
      if (isnan(v)||isinf(v)) {
        v = -log(a*universal_distribution(d));
        if (isnan(v)||isinf(v)) v = 1e8;
      } else {
        J(2*j+0) += (pred-d+bg)/var*gau/prob;
        J(2*j+1) += 0.5*(1/var-pow((pred-d+bg)/var,2)/2)*gau/prob;
      }
      value+=v;
      debug+=v;
      for (auto it1 =get<2>(background[i]).begin();
                it1!=get<2>(background[i]).end();
              ++it1) {
        const size_t k = reverse_index[*it1];
        const double _bg = x(2*k+0);
        const double _bg_var = abs(x(2*k+1));
        const double dist = 16;
        const double var = (bg_var+_bg_var)/dist;
        double v = 0.5*(pow(bg-_bg,2)/var+log(2*pi*var));
        if (isnan(v)||isinf(v)) {
          v = 1e8;
          value += 1e8;
        } else {
          value += v;
          J(2*j+0)  += (bg-_bg)/var;
          J(2*j+1)  +=
            (1/(bg_var+_bg_var)-16*pow((bg-_bg)/(bg_var+_bg_var),2))/2;
          J(2*k+0)  += (_bg-bg)/var;
          J(2*k+1)  +=
            (1/(bg_var+_bg_var)-16*pow((bg-_bg)/(bg_var+_bg_var),2))/2;
        }
      }
    }
    cerr << value << " " << debug << endl;
    return {value,J};
  }
};

void test_background_target(
    const background_target& target,
    const matrix<double,0,1>& x
    )
{
  const matrix<double,0,1> J = get<1>(target(x));
  const double epsilon = 10;
  for (size_t i=0;i!=x.nr();++i) {
    matrix<double,0,1> p = x;
    p(i)+=epsilon;
    matrix<double,0,1> m = x;
    m(i)-=epsilon;
    cout << J(i) << " "
         << 0.5*(get<0>(target(p))-get<0>(target(m)))/epsilon << endl; 
  }
}

struct const_background_functor{
  const double * background;
  const double * variance;
  const tuple<double,double> operator()(const size_t& i) const {
    return {background[i],variance[i]};
  }
}

struct background_functor{
  const double* data;
  const double& a;
  const double& g;
  const geometry::geometry& geom;
  const struct source& source;
  vector<tuple<double,double,double>> radial_median;
  patchmap<size_t,tuple<double,double,vector<size_t>>> background;
  background_functor(
      const double& a,
      const double& g,
      const geometry::geometry& geom,
      const struct source& source,
      const double* data,
      const uint8_t* mask
      ) :a(a),g(g),geom(geom),source(source),data(data) {
    cerr << "background functor constructor" << endl;
    // guess background, radial median
    vector<tuple<size_t,double,double>> pixels;
    for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
      for (size_t ss=0;ss!=it->nss;++ss) {
        for (size_t fs=0;fs!=it->nfs;++fs) {
          const size_t i = (*it)(fs,ss);
          if (mask[i]) continue;
          matrix<double,2,1> fsss{0.5+fs,0.5+ss};
          const matrix<double,3,1> x = normalize((*it)(fsss));
          pixels.emplace_back(i,trans(x)*normalize(source.kin),data[i]);
        }
      }
    }
    sort(pixels.begin(),pixels.end(),
        [](const auto& lhs,const auto& rhs)
        {return get<1>(lhs)<get<1>(rhs);}
        );
    //cerr << "sorted " << pixels.size() << " pixels"<< endl;
    const size_t m = 1024;
    size_t n = pixels.size()/m;
    radial_median.reserve(n);
    for (size_t i=0;i<pixels.size();i+=m) {
      nth_element(&pixels[i],&pixels[i+m/2],&pixels[i+m],
          [](const auto& lhs,const auto& rhs)
          {return get<2>(lhs)<get<2>(rhs);});
      double sumw=0,mean=0,M2=0;
      for (size_t j=i;j!=i+m;++j)
        mean_variance(get<2>(pixels[j]),1.0,sumw,mean,M2);
      //cerr << i << " " << get<2>(pixels[i+m/2]) << " " << sqrt(M2/sumw) << endl;
      radial_median.emplace_back(
          get<1>(pixels[i+m/2]),
          get<2>(pixels[i+m/2]),
          M2/sumw);
      //cerr << get<0>(radial_median.back()) << " " 
      //     << get<1>(radial_median.back()) << " " 
      //     << get<2>(radial_median.back()) << endl;
    } 
  }
  const tuple<double,double> operator()(const size_t& i) const {
    //cerr << "background_functor::operator()" << endl;
    //cerr << "cached pixels: " << background.size() << endl;
    //cerr << background.count(i) << endl;
    if (background.count(i)) {
      //cerr << i << " "
      //     << get<0>(background.at(i)) << " "
      //     << get<1>(background.at(i)) << endl;
      return {get<0>(background.at(i)),get<1>(background.at(i))};
    } else {
      //cerr << "fallback" << endl;
      const auto panel = geom.get_panel(i);
      const auto [fs,ss] = panel(i);
      const double d = trans(normalize(source.kin))
        *normalize(panel(matrix<double,2,1>{fs+0.5,ss+0.5}));
      auto it = lower_bound(
          radial_median.begin(),
          radial_median.end(),
          d,
          [](const auto& lhs,const auto& rhs){return get<0>(lhs)<rhs;}
          );
      //cerr << d << " " << get<0>(*it) << " "
      //                 << get<1>(*it) << " "
      //                 << get<2>(*it) << endl; 
      if (it!=radial_median.begin())
        if (abs(get<0>(*(it-1))-d)<abs(get<0>(*it)-d)) --it;
      //cerr << d << " " << get<0>(*it) << " "
      //                 << get<1>(*it) << " "
      //                 << get<2>(*it) << endl; 
      return {data[i],2*(pow(g,2)+get<2>(*it))};
      return {get<1>(*it),2*(pow(g,2)+get<2>(*it))};
    }
  }
  void apply_mask(
      const patchmap<size_t,void>& pixels
      ) {
    for (auto it=background.begin();it!=background.end();) {
      if (pixels.count(it->first)) {
        get<2>(it->second).erase(
            remove_if(
              get<2>(it->second).begin(),
              get<2>(it->second).end(),
              [&pixels = as_const(pixels)](const size_t& i)
              {return pixels.count(i)==0;}
              ),
            get<2>(it->second).end()
            );
        for (auto it1 =get<2>(it->second).begin();
                  it1!=get<2>(it->second).end();
                ++it1) {
          if (background.count(*it1)==0)
            cout << "this should not happen (1)" << endl;
        }
        ++it;
      } else {
        it=background.erase(it);
      }
    }
    for (auto it=pixels.begin();it!=pixels.end();++it) {
      const size_t i = *it;
      if (background.count(i)) continue;
      const auto panel = geom.get_panel(i);
      const auto [fs,ss] = panel(i);
      const auto [_bg,_var] = (*this)(i);
      auto& [bg,var,neighbours] = background[i];
      bg = _bg; var = _var;
      //cerr << i << " "
      //     << get<0>(background.at(i)) << " "
      //     << get<1>(background.at(i)) << " "
      //     << _bg << " " << _sigma << endl;
      for (int64_t iss = int64_t(ss)-1;iss<=int64_t(ss)+1;++iss) {
        for (int64_t ifs = int64_t(fs)-1;ifs<=int64_t(fs)+1;++ifs) {
          if (!panel.isvalid(ifs,iss)) continue;
          const double j = panel(ifs,iss);
          if (j==i) continue;
          if (background.count(j)==0) continue;
          get<2>(background[j]).push_back(i);
          get<2>(background[i]).push_back(j);
        }
      }
    }
    for (auto it=background.begin();it!=background.end();++it) {
      if (get<2>(it->second).size()>8) {
        cout << "this should not happen" << endl;
        cout << get<2>(it->second).size() << endl; 
      }
    }
  }
  const void optimise_(
      const patchmap<size_t,tuple<double,double>>& rendering
      ) {
    cerr << "starting background_functor::optimise" << endl;
    for (size_t i=0;i!=16;++i) {
      cerr << setw(8) << i << endl;
      for (auto it0=background.begin();it0!=background.end();++it0) {
        if (rendering.count(it0->first)==0) continue;
        vector<tuple<double,double>> neighbours;
        neighbours.reserve(get<2>(it0->second).size());
        for (auto it1 =get<2>(it0->second).begin();
                  it1!=get<2>(it0->second).end();
                ++it1) {
          neighbours.emplace_back(
              get<0>(background[*it1]),
              get<1>(background[*it1])
              );
        }
        individual_background_target target{
          a,
          data[it0->first],
          get<0>(rendering[it0->first]),
          get<1>(rendering[it0->first]),
          neighbours
        };
        matrix<double,2,1> x{get<0>(it0->second),abs(get<1>(it0->second))};
        find_min_gradient(x,target,count_stop_strategy{64,64},g*1e-4);
        get<0>(it0->second) = x(0);
        get<1>(it0->second) = abs(x(1));
      }
    }
  }
  const void optimise(
      const patchmap<size_t,tuple<double,double>>& rendering
      ) {
    cerr << "starting background_functor::optimise" << endl;
    background_target target{
      a,
      data,
      rendering,
      background,
      {}
    };
    cerr << background.size() << endl;
    matrix<double,0,1> x(2*background.size(),1);
    size_t j = 0;
    for (auto it=background.begin();it!=background.end();++it) {
      x(2*j+0) = get<0>(it->second);
      x(2*j+1) = abs(get<1>(it->second));
      target.reverse_index[it->first] = j;
      ++j;
    }
    //test_background_target(target,x);
    j = 0;
    find_min_gradient(x,target,count_stop_strategy{256,256},g*1e-4);
    for (auto it=background.begin();it!=background.end();++it) {
      get<0>(it->second) = x(2*j+0);
      get<1>(it->second) = abs(x(2*j+1));
      ++j;
    }
  }
  void print() const {
    for (auto it=background.begin();it!=background.end();++it) {
      const auto panel = geom.get_panel(it->first);
      const auto [fs,ss] = panel(it->first);
      const matrix<double,3,1> x = panel(matrix<double,2,1>{fs+0.5,ss+0.5});
      cout << x(0) << " " << x(1) << " " << x(2) << " "
           << get<0>(it->second) << " " << get<1>(it->second) << endl;
    }
  }
};

template<class T>
const
patchmap<size_t,tuple<double,double>> render_foreground(
  const T& pred,
  const double& g,
  const double& b,
  const double& c,
  const patchmap<IDX,double>& intensities,
  const patchmap<size_t,void>& pixel_mask
    ) {
  patchmap<size_t,tuple<double,double>> rendering;
  for (const auto& i : pixel_mask) rendering[i]={0.0,0.0};
  for (const auto & [hkl,pixels] : pred) {
    if (!intensities.count(hkl)) continue;
    const double& intensity = intensities.at(hkl);
    for (const auto & [i,_flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
      if (!pixel_mask.count(i)) continue;
      const double flx = intensity*_flx;
      get<0>(rendering[i])+=flx*wvn*g;
      get<1>(rendering[i])+=pow(g,2)*(b+c*flx)*flx*(pow(wvn,2)+pow(bnd,2));
    }
  }
  return rendering;
}

template<class F>
const void render_background(
  patchmap<size_t,tuple<double,double>>& rendering,
  const F& background
    ) {
  for (auto it=rendering.begin();it!=rendering.end();++it) {
    const auto [bg,bg_var] = background(it->first);
    get<0>(it->second) += bg;
    get<1>(it->second) += bg_var;
  }
}

template<class T,class F>
const
patchmap<size_t,tuple<double,double>> render(
  const T& pred,
  const double& g,
  const double& b,
  const double& c,
  const patchmap<IDX,double>& intensities,
  const F& background,
  const patchmap<size_t,void>& pixel_mask
    ) {
  patchmap<size_t,tuple<double,double>> rendering =
    render_foreground(pred,g,b,c,intensities,pixel_mask);
  render_background(rendering,background);
  return rendering;
}

struct crystal_target{
  const double* data;
  const patchmap<size_t,void>& pixel_mask;
  const background_functor& background;
  const patchmap<IDX,double>& intensities;
  const vector<source>& sources;
  const double& g;
  const double& a;
  const geometry::geometry geom;
  vector<IDX> indices;
  crystal_target(
      const double* data,
      const patchmap<size_t,void>& pixel_mask,
      const background_functor& background,
      const patchmap<IDX,double>& intensities,
      const vector<source>& sources,
      const double& g,
      const double& a,
      const geometry::geometry geom
    ) :
    data(data),
    pixel_mask(pixel_mask),
    background(background),
    intensities(intensities),
    sources(sources),
    g(g),
    a(a),
    geom(geom)
  {
    for (auto it=intensities.begin();it!=intensities.end();++it)
      indices.push_back(it->first);
  }
  const tuple<double,matrix<double,21,1>>
  operator()(const matrix<double,21,1>& x) const {
    struct crystl crystl;
    double b,c;
    set_crystal_from_vector(x,crystl,b,c);
    auto prediction = predict(sources,crystl,geom,indices,exp(-1));
    const auto rendering =
      render(prediction,g,b,c,intensities,background,pixel_mask);
    const double value = common_target(a,data,rendering);
    matrix<double,21,1> J = zeros_matrix<double>(21,1);
    for (const auto & [hkl,pixels] : prediction) {
      if (!intensities.count(hkl)) continue;
      const double intensity = intensities.at(hkl);
      for (const auto & [i,_flx,wvn,bnd,_dflx,_dwvn,_dbnd] : pixels) {
        if (!rendering.count(i)) continue;
        const double flx = intensity*_flx;
        const double bg  = get<0>(rendering.at(i))
                         - flx*wvn*g;
        const double bg_sigma = sqrt(
              get<1>(rendering.at(i))
            - pow(g,2)*(b+c*flx)*flx*(pow(wvn,2)+pow(bnd,2))
            );
        const auto llg_nabla = llg_pixel(
            data[i],
            g,
            flx,
            wvn,
            bnd,
            bg,
            bg_sigma,
            a,
            b,
            c,
            universal_distribution
            );
        if (isnan(get< 0>(llg_nabla))
          ||isnan(get< 3>(llg_nabla))
          ||isnan(get< 4>(llg_nabla))
          ||isnan(get< 5>(llg_nabla))
          ||isnan(get< 9>(llg_nabla))
          ||isnan(get<10>(llg_nabla))) continue;
        const struct crystl dflx = _dflx*(intensity*get<3>(llg_nabla));
        const struct crystl dwvn = _dwvn*get<4>(llg_nabla);
        const struct crystl dbnd = _dbnd*get<5>(llg_nabla);
        J( 0) +=    dflx.R(0,0)    + dwvn.R(0,0)    + dbnd.R(0,0)    ;
        J( 1) +=    dflx.R(0,1)    + dwvn.R(0,1)    + dbnd.R(0,1)    ;
        J( 2) +=    dflx.R(0,2)    + dwvn.R(0,2)    + dbnd.R(0,2)    ;
        J( 3) +=    dflx.R(1,0)    + dwvn.R(1,0)    + dbnd.R(1,0)    ;
        J( 4) +=    dflx.R(1,1)    + dwvn.R(1,1)    + dbnd.R(1,1)    ;
        J( 5) +=    dflx.R(1,2)    + dwvn.R(1,2)    + dbnd.R(1,2)    ;
        J( 6) +=    dflx.R(2,0)    + dwvn.R(2,0)    + dbnd.R(2,0)    ;
        J( 7) +=    dflx.R(2,1)    + dwvn.R(2,1)    + dbnd.R(2,1)    ;
        J( 8) +=    dflx.R(2,2)    + dwvn.R(2,2)    + dbnd.R(2,2)    ;
        J( 9) +=    dflx.mosaicity + dwvn.mosaicity + dbnd.mosaicity ;
        J(10) +=    dflx.peak(0,0) + dwvn.peak(0,0) + dbnd.peak(0,0) ;
        J(11) += 2*(dflx.peak(0,1) + dwvn.peak(0,1) + dbnd.peak(0,1));
        J(12) += 2*(dflx.peak(0,2) + dwvn.peak(0,2) + dbnd.peak(0,2));
        J(13) +=    dflx.peak(1,1) + dwvn.peak(1,1) + dbnd.peak(1,1) ;
        J(14) += 2*(dflx.peak(1,2) + dwvn.peak(1,2) + dbnd.peak(1,2));
        J(15) +=    dflx.peak(2,2) + dwvn.peak(2,2) + dbnd.peak(2,2) ;
        J(16) +=    dflx.strain    + dwvn.strain    + dbnd.strain    ;
        J(17) +=    dflx.a         + dwvn.a         + dbnd.a         ;
        J(18) +=    dflx.b         + dwvn.b         + dbnd.b         ;
        J(19) += get< 9>(llg_nabla);
        J(20) += get<10>(llg_nabla);
      }
    }
    cerr << value << endl;
    //cerr << trans(x) << endl;
    //cerr << trans(J) << endl;
    return {value,J};
  }
};

struct individual_foreground_target{
  const double& g;
  const double& a;
  const double& b;
  const double& c;
  // x flx wvn bnd bg var_bg
  vector<tuple<double,double,double,double,double,double>> pixels;
  template<typename T>
  individual_foreground_target(
      const double& g,
      const double& a,
      const double& b,
      const double& c,
      const double* data,
      const uint8_t* mask,
      const T& pred,
      const patchmap<IDX,double>& intensities,
      const patchmap<size_t,tuple<double,double>>& rendering,
      const IDX& index
      ):g(g),a(a),b(b),c(c)
  {
    for (const auto & [hkl,entry] : pred) {
      if (hkl!=index) continue;
      const double& intensity = intensities.count(hkl)?intensities.at(hkl):0.0;
      for (const auto & [i,_flx,wvn,bnd,dflx,dwvn,dbnd] : entry) {
        if (mask[i]) continue;
        if (rendering.count(i)==0) continue;
        const double flx = _flx*intensity;
        const auto& [_bg,_var_bg] = rendering.at(i);
        const double bg = _bg-flx*wvn*g;
        const double var_bg =
          _var_bg-pow(g,2)*(b+c*flx)*flx*(pow(wvn,2)+pow(bnd,2));
        pixels.emplace_back(data[i],_flx,wvn,bnd,bg,var_bg);
      }
    }
  }
  const tuple<double,matrix<double,1,1>>
  operator()(const matrix<double,1,1>& x) const {
    double value = 0;
    matrix<double,1,1> J = zeros_matrix<double>(1,1);
    for (auto it=pixels.begin();it!=pixels.end();++it) {
      const auto& [d,flx,wvn,bnd,bg,var_bg] = *it;
      const auto tmp = llg_pixel(
          d,
          g,
          flx*x(0)*(x(0)>0),
          wvn,
          bnd,
          bg,
          sqrt(var_bg),
          a,
          b,
          c,
          universal_distribution
          );
      value += get<0>(tmp);
      J(0)  += (x(0)>0)*flx*get<3>(tmp);
    }
    value -= x(0)*(x(0)<0);
    J(0)  -= (x(0)<0);
    //return {value,J};
    //value += abs(x(0));
    //J(0)  += signum(x(0));
    value += (0.5*(pow(x(0)-1,2)+log(2*pi)))/16.0;
    J(0)  += (x(0)-1)/16.0;
    return {value,J};
  }
};

template<background_functor>
struct foreground_target{
  const double* data;
  const double& g;
  const double& a;
  const double& b;
  const double& c;
  const patchmap<size_t,void>& pixel_mask;
  const background_functor& background;
  const vector<source>& sources;
  const struct crystl& crystl;
  const geometry::geometry geom;
  patchmap<IDX,double> intensities;
  patchmap<IDX,size_t> reverse_index;
  vector<IDX> indices;
  foreground_target(
    const double* data,
    const double& g,
    const double& a,
    const double& b,
    const double& c,
    const patchmap<size_t,void>& pixel_mask,
    const background_functor& background,
    const vector<source>& sources,
    const struct crystl& crystl,
    const geometry::geometry geom,
    patchmap<IDX,double> intensities
      ) :
    data(data),
    g(g),a(a),b(b),c(c),
    pixel_mask(pixel_mask),
    background(background),
    sources(sources),
    crystl(crystl),
    geom(geom),
    intensities(intensities)
  {
    size_t i = 0;
    for (auto it=intensities.begin();it!=intensities.end();++it) {
      reverse_index[it->first] = i;
      indices.push_back(it->first);
      ++i;
    }
    indices.shrink_to_fit();
  }
  const tuple<double,matrix<double,0,1>>
  operator()(const matrix<double,0,1>& x) const {
    {
      size_t i = 0;
      for (auto it=intensities.begin();it!=intensities.end();++it) {
        it->second = x(i)>0?x(i):0;
        ++i;
      }
    }
    matrix<double,0,1> J = zeros_matrix<double>(x.nr(),1);
    const auto prediction = predict(sources,crystl,geom,indices,exp(-1));
    const auto rendering =
      render(prediction,g,b,c,intensities,background,pixel_mask);
    double value = common_target(a,data,rendering);
    for (const auto & [hkl,pixels] : prediction) {
      if (!intensities.count(hkl)) continue;
      const double& intensity = intensities.at(hkl);
      for (const auto & [i,_flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
        if (rendering.count(i)==0) continue;
        const double flx = intensity*_flx;
        const double bg = get<0>(rendering.at(i))-flx*wvn*g;
        const double var_bg =
          get<1>(rendering.at(i))-pow(g,2)*(b+c*flx)*flx*(pow(wvn,2)+pow(bnd,2));
        const double d = data[i];
        const auto tmp = llg_pixel(
            d,
            g,
            flx,
            wvn,
            bnd,
            bg,
            sqrt(var_bg),
            a,
            b,
            c,
            universal_distribution
            );
        J(reverse_index.at(hkl)) += _flx*(isnan(get<3>(tmp))?0:get<3>(tmp));
      }
    }
    for (size_t i=0;i!=x.nr();++i) {
      value -= x(i)*(x(i)<0);
      J(i) -= (x(i)<0);
    }
    //cerr << value << endl;
    //return {value,J};
    cerr << value << " ";
    for (size_t i=0;i!=x.nr();++i) {
      value += (0.5*(pow(x(i)-1,2)+log(2*pi)))/16.0;
      J(i) += (x(i)-1)/16.0;
    }
    cerr << value << endl;
    return {value,J};
  }
};

void test_foreground_target(
    const foreground_target& target,
    const matrix<double,0,1>& x
    )
{
  const double epsilon = 1e-8;
  const matrix<double,0,1> J = get<1>(target(x));
  for (size_t i=0;i!=x.nr();++i) {
    matrix<double,0,1> xp = x;
    xp(i)+=epsilon;
    matrix<double,0,1> xm = x;
    xm(i)-=epsilon;
    cerr << J(i) << " "
         << 0.5*(get<0>(target(xp))-get<0>(target(xm)))/epsilon << endl;
  }
}

/*struct background_target{
  const double& x;
  const double& g;
  const double& flx;
  const double& wvn;
  const double& bnd;
  const double& a;
  const double& b;
  const double& c;
  const vector<tuple<double,double>> neighbours;
  const tuple<double,matrix<double,2,1>>
  operator(const matrix<double,2,1>& x) const {
    double value = 0;
    matrix<double,2,1> J = zeros_matrix<double>(2,1);
    const auto tmp = llg_pixel(
        x,
        g,
        flx,
        wvn,
        bnd,
        x(0),
        x(1),
        a,
        b,
        c,
        universal_distribution
        );
    value+=get<0>(tmp);
    J(0) +=get<6>(tmp);
    J(1) +=get<7>(tmp);
    for (auto it=neighbours.begin();it!=neighbours.end();++it) {
      const double dist = 9;
      const double var = pow(sigma_bg/dist,2)+pow(get<1>(*it)/dist,2);
      value+=0.5*(pow(bg-get<0>(*it),2)/var-log(2*pi*var));
      J(0)+=(bg-get<0>(*it))/var;
      J(1)-=sigma_bg*(pow((bg-get<0>(*it))/var,2)+1/var)/pow(dist,2);
    }
    return {value,J};
  }
};*/

const patchmap<size_t,void> expand_mask(
    const patchmap<size_t,void>& pixels,
    const geometry::geometry& geom,
    const uint8_t* mask,
    const double& radius = 2.5
    ) {
  patchmap<size_t,void> expanded;
  for (auto it=pixels.begin();it!=pixels.end();++it) {
    const size_t i = *it;
    const auto panel = geom.get_panel(i);
    const auto [fs,ss] = panel(i);
    for (int64_t iss=int64_t(ss)-6;iss<=int64_t(ss)+6;++iss) {
      for (int64_t ifs=int64_t(fs)-6;ifs<=int64_t(fs)+6;++ifs) {
        if (sqrt(pow(int64_t(fs)-ifs,2)+pow(int64_t(ss)-iss,2))>radius)
          continue;
        if (!panel.isvalid(ifs,iss)) continue;
        const size_t j = panel(ifs,iss);
        if (mask[j]) continue;
        expanded.emplace(j);
      }
    }
  }
  return expanded;
}

const patchmap<size_t,double> insert_data_to_mask(
    const patchmap<size_t,void>& mask,
    double* data
    ) {
  patchmap<size_t,double> pixels(mask.size());
  for (auto it=mask.begin();it!=mask.end();++it) {
    pixels.emplace(*it,data[*it]);
  }
  return pixels;
}

void test_crystl_target(
    const matrix<double,21,1>& x,
    const crystal_target& target
    ){
  const auto result = target(x);
  for (size_t i=0;i!=21;++i) {
    cout << setw(16) << get<1>(result)(i);
    for (double epsilon = 1e-7;epsilon>2e-11;epsilon*=1e-1) {
      matrix<double,21,1> xp = x;
      xp(i)+=epsilon;
      matrix<double,21,1> xm = x;
      xm(i)-=epsilon;
      cout << setw(16) << 0.5*(get<0>(target(xp))-get<0>(target(xm)))/epsilon;
    }
    cout << endl;
  }
  cout << endl;
}

void test_pred_der(
    const matrix<double,21,1>& x,
    const vector<struct source>& sources,
    const geometry::geometry& geom
    ) {
  double b=0,c=0;
  struct crystl crystl;
  patchmap<tuple<IDX,size_t,size_t,int>,tuple<double,double,double>> data;
  const double epsilon = 2e-9;
  for (size_t j=0;j!=19;++j) {
    cerr << j << endl;
    matrix<double,21,1> xp = x;
    xp(j)+=epsilon;
    set_crystal_from_vector(xp,crystl,b,c);
    auto prediction = predict(sources,crystl,geom,exp(-1),2);
    for (const auto & [hkl,pixels] : prediction) {
      for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
        data[{hkl,i,j,1}] = {flx,wvn,bnd};
      }
    }    
    matrix<double,21,1> xm = x;
    xm(j)-=epsilon;
    set_crystal_from_vector(xm,crystl,b,c);
    prediction = predict(sources,crystl,geom,exp(-1),2);
    for (const auto & [hkl,pixels] : prediction) {
      for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
        data[{hkl,i,j,-1}] = {flx,wvn,bnd};
      }
    }
  }
  set_crystal_from_vector(x,crystl,b,c);
  const auto prediction = predict(sources,crystl,geom,exp(-1),2);
  for (const auto & [hkl,pixels] : prediction) {
    cout << get<0>(hkl) << " " << get<1>(hkl) << " " << get<2>(hkl) << endl;
    for (const auto & [i,flx,wvn,bnd,_dflx,_dwvn,_dbnd] : pixels) {
      cout << i << endl;
      matrix<double,21,1> dflx;
      set_crystal_vector(dflx,_dflx,b,c);
      matrix<double,21,1> dwvn;
      set_crystal_vector(dwvn,_dwvn,b,c);
      matrix<double,21,1> dbnd;
      set_crystal_vector(dbnd,_dbnd,b,c);
      for (size_t j=0;j!=19;++j) {
        cout << setw(16) << dflx(j); 
        if (data.count({hkl,i,j,1})&&data.count({hkl,i,j,-1})) {
          cout << setw(16)
               << (get<0>(data[{hkl,i,j,1}])-get<0>(data[{hkl,i,j,-1}]))
                  *0.5/epsilon;
        } else {
          cout << setw(16) << "?";
        }
        cout << setw(16) << dwvn(j); 
        if (data.count({hkl,i,j,1})&&data.count({hkl,i,j,-1})) {
          cout << setw(16)
               << (get<1>(data[{hkl,i,j,1}])-get<1>(data[{hkl,i,j,-1}]))
                  *0.5/epsilon;
        } else {
          cout << setw(16) << "?";
        }
        cout << setw(16) << dbnd(j); 
        if (data.count({hkl,i,j,1})&&data.count({hkl,i,j,-1})) {
          cout << setw(16)
               << (get<2>(data[{hkl,i,j,1}])-get<2>(data[{hkl,i,j,-1}]))
                  *0.5/epsilon;
        } else {
          cout << setw(16) << "?";
        }
      }
      cout << endl;
    }
  }
}

geometry::geometry optimise_geometry(
    const double* data,
    const double& g,
    const double& a,
    const double& b,
    const double& c,
    const vector<struct source>& sources,
    const struct crystl& crystl,
    const geometry::geometry& geom,
    const patchmap<size_t,void>& pixel_mask,
    const patchmap<IDX,double>& intensities,
    const background_functor& background
    ) {
  vector<IDX> indices;
  for (auto it=intensities.begin();it!=intensities.end();++it)
    indices.push_back(it->first);
  geometry::geometry best_geom = geom;
  double best_value =
    common_target(
        a,data,
        render(
          predict(
            sources,crystl,geom,indices,exp(-1)
          )
          ,g,b,c,intensities,background,pixel_mask
        )
    );
  random_device rd;
  normal_distribution<double> nd(0,1);
  for (size_t i=0;i!=2;++i) {
    geometry::geometry geom_copy = best_geom;
    for (auto it=geom_copy.panels.begin();it!=geom_copy.panels.end();++it) {
      (*it).o(2)+=nd(rd);
    }
    const double value =
        common_target(
            a,data,
            render(
              predict(
                sources,crystl,geom_copy,indices,exp(-1)
              )
              ,g,b,c,intensities,background,pixel_mask
            )
        );
    if (value<best_value) {
      best_geom = geom_copy;
      best_value = value;
      cerr << i << " " << value << endl;
    }
  }
  for (size_t i=0;i!=geom.panels.size();++i) {
    geometry::geometry geom_copy = best_geom;
    geom_copy.panels[i%geom_copy.panels.size()].o(0)+=nd(rd);
    geom_copy.panels[i%geom_copy.panels.size()].o(1)+=nd(rd);
    geom_copy.panels[i%geom_copy.panels.size()].D(0,0)+=nd(rd)/1000;
    geom_copy.panels[i%geom_copy.panels.size()].D(0,1)+=nd(rd)/1000;
    geom_copy.panels[i%geom_copy.panels.size()].D(1,0)+=nd(rd)/1000;
    geom_copy.panels[i%geom_copy.panels.size()].D(1,1)+=nd(rd)/1000;
    const double value =
        common_target(
            a,data,
            render(
              predict(
                sources,crystl,geom_copy,indices,exp(-1)
              )
              ,g,b,c,intensities,background,pixel_mask
            )
        );
    if (value<best_value) {
      best_geom = geom_copy;
      best_value = value;
      cerr << i << " " << value << endl;
    }
  }
  return best_geom;
}

void print_panelpositions(
    const geometry::geometry& geom
    ) {
  size_t i = 0;
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    const matrix<double,3,1> x0 =
      (*it)(matrix<double,2,1>{0.0        ,0.0        });
    const matrix<double,3,1> x1 =
      (*it)(matrix<double,2,1>{1.0*it->nfs,0.0        });
    const matrix<double,3,1> x2 =
      (*it)(matrix<double,2,1>{1.0*it->nfs,1.0*it->nss});
    const matrix<double,3,1> x3 =
      (*it)(matrix<double,2,1>{0.0        ,1.0*it->nss});
    const matrix<double,3,1> c =
      (*it)(matrix<double,2,1>{0.5*it->nfs,0.5*it->nss});
    cout << x0(0) << " " << x0(1) << " " << x0(2) << endl;
    cout << x1(0) << " " << x1(1) << " " << x1(2) << endl;
    cout << x2(0) << " " << x2(1) << " " << x2(2) << endl;
    cout << x3(0) << " " << x3(1) << " " << x3(2) << endl;
    cout << x0(0) << " " << x0(1) << " " << x0(2) << endl;
    cout <<  c(0) << " " <<  c(1) << " " <<  c(2) << " " << i << endl;
    ++i;
  }
}

template<background_functor>
void save_rendering(
    const patchmap<size_t,tuple<double,double>>& rendering,
    const background_functor& background,
    const geometry::geometry& geom,
    const double* data,
    const uint8_t* mask
    ) {
  ofstream tmp("rendering");
  for (auto it=rendering.begin();it!=rendering.end();++it) {
    const auto [fs,ss] = geom.get_panel(it->first)(it->first);
    matrix<double,3,1> xyz = geom.get_panel(it->first)(
        matrix<double,2,1>{1.0*fs,1.0*ss});
    tmp << xyz(0) << " " << xyz(1) << " " << xyz(2) << " "
        << get<0>(it->second)-get<0>(background(it->first)) << " "
        << get<0>(background(it->first)) << " "
        << get<1>(it->second)-get<1>(background(it->first)) << " " 
        << get<1>(background(it->first)) << " "
        << data[it->first] << endl;
  }
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    ofstream file("panel"+to_string(it-geom.panels.begin())+".pgm");
    file << "P5 " << it->nfs << " " << it->nss << " " << 255 << endl;
    for (size_t ss=0;ss!=it->nss;++ss) for (size_t fs=0;fs!=it->nfs;++fs) {
          const size_t i = (*it)(fs,ss);
          const uint8_t n =
            clip(4*((mask[i]==0)*log2(1+pow(size_t(data[i]),2))),0,255);
          file.put(n);
    }
  }
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    ofstream file("panel_"+to_string(it-geom.panels.begin())+".pgm");
    file << "P5 " << it->nfs << " " << it->nss << " " << 255 << endl;
    for (size_t ss=0;ss!=it->nss;++ss) for (size_t fs=0;fs!=it->nfs;++fs) {
          const size_t i = (*it)(fs,ss);
          const double v = rendering.count(i)?
                           get<0>(rendering[i]):
                           get<0>(background(i));
          const uint8_t n =
            clip(4*(mask[i]==0)*log2(1+pow(size_t(v),2)),0,255);
          file.put(n);
    }
  }
}

int main() {
  const double a = 1.0/4096;//1.0/256;
  const auto g = 9511.11*0.00105;
  double b = 1.0;
  double c = 1.0/16;
  cerr << setprecision(15);
  auto sources = deserialize_sources(cin);
  if (!cin) {
    cerr << "error reading sources" << endl;
    return 1;
  }
  cerr << "read sources" << endl;
  for (auto it=sources.begin();it!=sources.end();++it) {
    cerr << "< " << it->flx << " "
      << it->kin(0) << " " << it->kin(1) << " " << it->kin(2) << endl
      << it->S12(0,0) << " " << it->S12(0,1) << " " << it->S12(0,2) << endl
                             << it->S12(1,1) << " " << it->S12(1,2) << endl
                                                    << it->S12(2,2) << endl;
  }
  auto crystls = deserialize_crystls(cin);
  if (!cin) {
    cerr << "error reading sources" << endl;
    return 2;
  }
  //crystls.front().R*=matrix<double,3,3>{-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0};
  //crystls.front().U*=matrix<double,3,3>{-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0};
  cerr << "read crystals" << endl;
  for (auto it=crystls.begin();it!=crystls.end();++it) {
    cerr << it->R << endl;
    cerr << it->peak << endl;
    cerr << it->mosaicity << " " << it->strain << endl;
    cerr << it->a << " " << it->b << endl;
  }
  geometry::geometry geom;
  if (!read_geometry_bin(cin,geom)) {
    cerr << "read error in binary geometry file" << endl;
    return 3;
  }
  if (!cin) {
    cerr << "error reading binary geometry file" << endl;
    return 4;
  }
  cerr << "read binary geometry file, number of pixels = " << geom.num_pixels
       << endl;
  //for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
  //  cerr << it->nfs << " " << it->nss << endl;
  //  cerr << it->D(0,0) << " " << it->D(0,1) << " " << it->o(0) << endl;
  //  cerr << it->D(1,0) << " " << it->D(1,1) << " " << it->o(1) << endl;
  //  cerr << it->D(2,0) << " " << it->D(2,1) << " " << it->o(2) << endl;
  //}
  uint8_t* mask = new uint8_t[geom.num_pixels];
  // read in mask data
  cin.read(reinterpret_cast<char*>(mask),geom.num_pixels*sizeof(uint8_t));
  if (!cin) {
    cerr << "ran out of buffer for mask" << endl;
    return 5;
  }
  double* data = new double[geom.num_pixels*sizeof(double)];
  // read in image data
  cin.read(reinterpret_cast<char*>(data),geom.num_pixels*sizeof(double));
  if (!cin) {
    cerr << "ran out of buffer for data" << endl;
    return 6;
  }
  double* background_precomputed = new double[geom.num_pixels*sizeof(double)];
  cin.read(reinterpret_cast<char*>(background_precomputed),n*sizeof(double));
  if (!cin) {
    cerr << "ran out of buffer for background" << endl; 
  }
  double* variance_precomputed = new double[geom.num_pixels*sizeof(double)];
  cin.read(reinterpret_cast<char*>(variance_precomputed),n*sizeof(double));
  if (!cin) {
    cerr << "ran out of buffer for variance" << endl; 
  }
  auto crystl = crystls.front();
  auto source = average_source(sources);
  patchmap<IDX,double> intensities;
  //background_functor background(a,g,geom,source,data,mask);
  const_background_functor
    background{background_precomputed,variance_precomputed};
  //return 0;
  // loop:
  //   fit intensities
  //   fit background
  //   fit crystal
  // output h,k,l,i,δ²,background,δ²_background
  if (false) {
    const auto prediction = predict(sources,crystl,geom,exp(-1),2);
    return 0;
  }
  /*if (false) {
    partiality::test_predict(source,crystl);
    return 0;
  }*/
  if (false) {
    matrix<double,21,1> x;
    set_crystal_vector(x,crystl,b,c);
    test_pred_der(x,sources,geom);
    return 0;
  }
  for (size_t i=0;i!=256;++i) {
    cerr << '\r' << setw(16) << i;
    auto prediction = predict(sources,crystl,geom,exp(-1),2);
    cerr << "predicting " << prediction.size() << " indices" << endl;
    patchmap<size_t,void> pixel_mask;
    ofstream pixelmaskfile("pixelmask");
    for (const auto & [hkl,pixels] : prediction) {
      //cerr << get<0>(hkl) << " " << get<1>(hkl) << " " << get<2>(hkl) << endl;
      for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
        if (!mask[i]) pixel_mask.emplace(i);
        const auto [fs,ss] = geom.get_panel(i)(i);
        matrix<double,3,1> xyz = geom.get_panel(i)(
            matrix<double,2,1>{1.0*fs,1.0*ss});
        pixelmaskfile << trans(xyz);
      }
    }
    cerr << "finding relevant pixels" << endl;
    pixel_mask = expand_mask(pixel_mask,geom,mask);
    cerr << "render foreground" << endl;
    for (const auto & [hkl,pixels] : prediction)
      if (intensities.count(hkl)==0) intensities[hkl]=0.0;
    {
      ofstream indicesfile("indices");
      size_t i=0;
      for (auto it=intensities.begin();it!=intensities.end();++it) {
        indicesfile << get<0>(it->first) << " "
                    << get<1>(it->first) << " "
                    << get<2>(it->first) << endl;
        ++i;
      }
    }
    //for (const auto & [hkl,pixels] : prediction) intensities[hkl]=1.0;
    auto rendering = render_foreground(prediction,g,b,c,intensities,pixel_mask);
    cerr << rendering.size() << " pixels are relevant" << endl;
    //cerr << "apply mask" << endl;
    //background.apply_mask(pixel_mask);
    // fit background
    //cerr << "fit background" << endl;
    //save_rendering(rendering,background,geom,data,mask);
    //background.optimise(rendering);
    render_background(rendering,background);
    save_rendering(rendering,background,geom,data,mask);
    if constexpr (true) {
    // fit intensities
    cerr << "fit intensities individually" << endl;
    for (const auto & [hkl,pixels] : prediction) {
      if (intensities.count(hkl)==0) continue;
      individual_foreground_target target(
          g,a,b,c,
          data,
          mask,
          prediction,
          intensities,
          rendering,
          hkl
          );
      matrix<double,1,1> x;
      x(0) = intensities[hkl];
      double best_value = get<0>(target(x));
      for (int j=-64;j!=64;++j) {
        matrix<double,1,1> p;
        p(0) = pow(pow(2.0,0.25),j);
        const double value = get<0>(target(p));
        if (value < best_value) {
          intensities[hkl] = p(0);
          best_value = value;
        }
      }
    }
    rendering = render(
        predict(sources,crystl,geom,exp(-1),2),
        g,b,c,intensities,background,pixel_mask);
    cerr << "fit intensities all at once" << endl;
    foreground_target intensities_target
      (data,g,a,b,c,pixel_mask,background,sources,crystl,geom,intensities);
    matrix<double,0,1> x(intensities.size(),1);
    {
      size_t i=0;
      for (auto it=intensities.begin();it!=intensities.end();++it) {
        x(i) = it->second;
        ++i;
      }
    }
    //test_foreground_target(intensities_target,x);
    //return 0;
    find_min_gradient
      (x,intensities_target,count_stop_strategy{4096,4096},1e-4);
    cerr << "optimisation done" << endl;
    {
      ofstream intensityfile("intensities");
      size_t i=0;
      for (auto it=intensities.begin();it!=intensities.end();++it) {
        it->second = x(i)>0?x(i):0;
        intensityfile << get<0>(it->first) << " "
                      << get<1>(it->first) << " "
                      << get<2>(it->first) << " "
                      << it->second << endl;
        ++i;
      }
    }
    }
    rendering = render_foreground(prediction,g,b,c,intensities,pixel_mask);
    //background.optimise(rendering);
    render_background(rendering,background);
    save_rendering(rendering,background,geom,data,mask);
    cerr << "fit crystal" << endl;
    // fit crystal
    matrix<double,21,1> x;
    set_crystal_vector(x,crystl,b,c);
    crystal_target
      target(data,pixel_mask,background,intensities,sources,g,a,geom);
    //test_crystl_target(x,target); return 0;
    if constexpr (false) {
      double best_value = get<0>(target(x));
      matrix<double,21,1> y = x;
      for (int i=-4;i!=4;++i) {
        for (int j=-4;j!=4;++j) {
          matrix<double,21,1> p = y;
          p(17)*= pow(pow(2.0,0.25),i);
          p(18)*= pow(pow(2.0,0.25),j);
          double value = get<0>(target(p));
          if (value<best_value) {
            x = p;
            best_value = value;
          }
        }
      }
      cerr << "result of a b gridsearch " << best_value << endl;
    }
    find_min(x,target,count_stop_strategy{64,64},1e-13);
    set_crystal_from_vector(x,crystl,b,c);
    {
      ofstream crystlfile("_crystl"+to_string(i));
      crystlfile << setprecision(8);
      crystlfile << ">" << endl
                 << crystl.R
                 << crystl.peak
                 << crystl.mosaicity << " "
                 << crystl.strain << " "
                 << crystl.a << " "
                 << crystl.b << endl;
    }
    rendering = render(
        predict(sources,crystl,geom,exp(-1),2),
        g,b,c,intensities,background,pixel_mask);
    save_rendering(rendering,background,geom,data,mask);
    if constexpr (true) {
      cerr << "optimise geometry" << endl;
      geom = optimise_geometry(
          data,g,a,b,c,sources,crystl,geom,pixel_mask,intensities,background);
      ofstream geomfile("_geom"+to_string(i));
      for (auto it=geom.panels.begin();it!=geom.panels.end();++it)
        geomfile << fixed << setprecision(8)
          << it->nfs << " " << it->nss << endl
          << it->D(0,0) << " " << it->D(0,1) << " " << it->o(0) << endl
          << it->D(1,0) << " " << it->D(1,1) << " " << it->o(1) << endl
          << it->D(2,0) << " " << it->D(2,1) << " " << it->o(2) << endl;
    }
  }
}
