#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include "partiality.hpp"
#include "geometry.hpp"
#include "wmath.hpp"
#include "wmath_optimisation.hpp"

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

using partiality::IDX;
using partiality::crystl;
using partiality::deserialize_crystls;
using partiality::deserialize_sources;
using partiality::predict;
using partiality::source;

using whash::patchmap;

using wmath::clip;
using wmath::mean_variance;
using wmath::signum;
using wmath::count_stop_strategy;
using wmath::universal_distribution;
using wmath::find_min_numerical;

constexpr double pi = 3.141592653589793;
const     double min_flx = exp(-4);
constexpr size_t oversampling = 2;

inline tuple<double,tuple<double,double,double>> gauss_sd
(
  const double x,
  const double m,
  const double s
)
{
  const double e = pow((x-m)/s,2);
        double t = exp(-0.5*e)/(sqrt(2*pi)*abs(s));
  if (isnan(t)||isinf(t)) t = 0;
  return
  {
    t,
    {
      (m-x)/pow(s,2)*t,
      (x-m)/pow(s,2)*t,
      t*(e-1)/s
    }
  };
}

inline tuple<double,tuple<double,double,double>> gauss
(
  const double x,
  const double m,
  const double v
)
{
  const double e = pow((x-m),2)/abs(v);
        double t = exp(-0.5*e)/sqrt(2*pi*abs(v));
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

// test is successful
void test_gauss() {
  const double e = 1e-8;
  cout << "test gauss" << endl;
  for (double x=-7;x<=7;x+=1) {
    for (double m=-3;m<=3;m+=1) {
      for (double v=-5;v<=5;v+=1) {
  const auto tmp = gauss(x,m,v);
  cout << get<0>(get<1>(tmp)) << " "
       << (get<0>(gauss(x+e,m,v))-get<0>(gauss(x-e,m,v)))/(2*e) << endl; 
  cout << get<1>(get<1>(tmp)) << " "
       << (get<0>(gauss(x,m+e,v))-get<0>(gauss(x,m-e,v)))/(2*e) << endl; 
  cout << get<2>(get<1>(tmp)) << " "
       << (get<0>(gauss(x,m,v+e))-get<0>(gauss(x,m,v-e)))/(2*e) << endl;
      }
    }
  }
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

// tested and works perfectly
void test_prediction_mv() {
  const double e = 1e-8;
  const double g = 7;
  const double b = 1.3;
  const double c = 0.7;
  cout << "test_prediction_mv()" << endl;
  for (double flx=-17;flx<=17;flx+=5) {
    for (double wvn=-13;wvn<=13;wvn+=4) {
      for (double bnd=-5;bnd<=5;bnd+=2) {
  const auto [mv,J] = prediction_mv(g,flx,wvn,bnd,b,c);
  cout << J(0,0) << " "
    << (get<0>(prediction_mv(g+e,flx,wvn,bnd,b,c))(0)
       -get<0>(prediction_mv(g-e,flx,wvn,bnd,b,c))(0))/(2*e) << endl;
  cout << J(1,0) << " "
    << (get<0>(prediction_mv(g,flx+e,wvn,bnd,b,c))(0)
       -get<0>(prediction_mv(g,flx-e,wvn,bnd,b,c))(0))/(2*e) << endl;
  cout << J(2,0) << " "
    << (get<0>(prediction_mv(g,flx,wvn+e,bnd,b,c))(0)
       -get<0>(prediction_mv(g,flx,wvn-e,bnd,b,c))(0))/(2*e) << endl;
  cout << J(3,0) << " "
    << (get<0>(prediction_mv(g,flx,wvn,bnd+e,b,c))(0)
       -get<0>(prediction_mv(g,flx,wvn,bnd-e,b,c))(0))/(2*e) << endl;
  cout << J(4,0) << " "
    << (get<0>(prediction_mv(g,flx,wvn,bnd,b+e,c))(0)
       -get<0>(prediction_mv(g,flx,wvn,bnd,b-e,c))(0))/(2*e) << endl;
  cout << J(5,0) << " "
    << (get<0>(prediction_mv(g,flx,wvn,bnd,b,c+e))(0)
       -get<0>(prediction_mv(g,flx,wvn,bnd,b,c-e))(0))/(2*e) << endl;
  cout << J(0,1) << " "
    << (get<0>(prediction_mv(g+e,flx,wvn,bnd,b,c))(1)
       -get<0>(prediction_mv(g-e,flx,wvn,bnd,b,c))(1))/(2*e) << endl;
  cout << J(1,1) << " "
    << (get<0>(prediction_mv(g,flx+e,wvn,bnd,b,c))(1)
       -get<0>(prediction_mv(g,flx-e,wvn,bnd,b,c))(1))/(2*e) << endl;
  cout << J(2,1) << " "
    << (get<0>(prediction_mv(g,flx,wvn+e,bnd,b,c))(1)
       -get<0>(prediction_mv(g,flx,wvn-e,bnd,b,c))(1))/(2*e) << endl;
  cout << J(3,1) << " "
    << (get<0>(prediction_mv(g,flx,wvn,bnd+e,b,c))(1)
       -get<0>(prediction_mv(g,flx,wvn,bnd-e,b,c))(1))/(2*e) << endl;
  cout << J(4,1) << " "
    << (get<0>(prediction_mv(g,flx,wvn,bnd,b+e,c))(1)
       -get<0>(prediction_mv(g,flx,wvn,bnd,b-e,c))(1))/(2*e) << endl;
  cout << J(5,1) << " "
    << (get<0>(prediction_mv(g,flx,wvn,bnd,b,c+e))(1)
       -get<0>(prediction_mv(g,flx,wvn,bnd,b,c-e))(1))/(2*e) << endl;      
      }
    }
  }
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
  const auto [p_g,dg]  = gauss(x,m,v+0.5);
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

// tested and perfect
void test_llg() {
  const double e = 1e-4;
  const double a = 1.0/17;
  const auto o = universal_distribution;
  cout << "test llg" << endl;
  for (double x=-11;x<=11;x+=3) {
    for (double m=-7;m<=7;m+=4) {
      for (double v=-3;v<=3;++v) {
  const auto [l,d] = llg(x,m,v,a,o);
  cout << get<0>(d) << " "
       << (get<0>(llg(x,m+e,v,a,o))-get<0>(llg(x,m-e,v,a,o)))/(2*e) << endl;
  cout << get<1>(d) << " "
       << (get<0>(llg(x,m,v+e,a,o))-get<0>(llg(x,m,v-e,a,o)))/(2*e) << endl;
  cout << get<2>(d) << " "
       << (get<0>(llg(x,m,v,a+e,o))-get<0>(llg(x,m,v,a-e,o)))/(2*e) << endl;
      }
    }
  }
}

double common_target(
    const double a,
    const double* data,
    const patchmap<size_t,tuple<double,double>> prediction
    ) {
  double value = 0;
  for (auto it=prediction.begin();it!=prediction.end();++it) {
    const auto [m,v] = it->second;  
    const double x = data[it->first];
    const auto [t,der] = llg(x,m,v,a,universal_distribution);
    value += t;
    //if (isnan(value)) {
    //  cerr << "nan encountered in common target" << endl;
    //  cerr << m << " " << v << endl;
    //  cerr << x << " " << m << " " << v << " " << a << " " << t;
    //  return 1e300;
    //}
  }
  return value;
}

template<long int nr>
tuple<double,matrix<double,nr,1>>
common_target(
    const double a,
    const double* data,
    const patchmap<
      size_t,
      tuple<
        matrix<double,2,1>, // m,v
        matrix<double,nr,2> // derivatives to be calculated
      >
    > prediction
    ) {
  double value = 0;
  matrix<double,nr,1> J = zeros_matrix<double>(nr,1);
  for (auto it=prediction.begin();it!=prediction.end();++it) {
    const auto [mv,pred_J] = it->second;  
    const double x = data[it->first];
    const auto [t,der] = llg(x,mv(0),mv(1),a,universal_distribution);
    value += t;
    matrix<double,2,1> dmds{get<0>(der),get<1>(der)};
    J+=pred_J*trans(dmds);
  }
  return {value,J};
}

void inline set_crystal_vector(
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
  const matrix<double,3,3> peak_squared = crystl.peak;
  cholesky_decomposition<matrix<double,3,3>> chol(peak_squared);
  const matrix<double,3,3> peak = chol.get_l();
  x(10) = peak(0,0);
  x(11) = peak(0,1);
  x(12) = peak(0,2);
  x(13) = peak(1,1);
  x(14) = peak(1,2);
  x(15) = peak(2,2);
  x(16) = crystl.strain;
  x(17) = crystl.a;
  x(18) = crystl.b;
  x(19) = b;
  x(20) = c;
}

void inline set_crystal_from_vector(
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
   crystl.peak(0,1) = x(11);
   crystl.peak(0,2) = x(12);
   crystl.peak(1,1) = x(13);
   crystl.peak(1,2) = x(14);
   crystl.peak(2,2) = x(15);
   crystl.peak = trans(crystl.peak)*crystl.peak;
   crystl.strain    = x(16);
   crystl.a         = x(17);
   crystl.b         = x(18);
   b                = x(19);
   c                = x(20);
}

struct background_target{
  const double& a; // outlier fraction
  const double* data; // data
  const patchmap<size_t,tuple<double,double>>& rendering;
  patchmap<
    size_t,
    tuple<
      double,
      double,
      vector<
        tuple<
          size_t,
          double
        >
      >
    >
  >& background;
  patchmap<size_t,size_t> reverse_index; // pixel -> index
  const tuple<double,matrix<double,0,1>>
  operator()(const matrix<double,0,1>& x) const {
    double value = 0;
    double without_constraint = 0;
    matrix<double,0,1> J = zeros_matrix<double>(x.nr(),1);
    for (auto it0=rendering.begin();it0!=rendering.end();++it0) {
      const size_t i        = it0->first;
      const auto [bg,var]   = it0->second;
      if (background.count(i)==0) continue;
      const long int j      = reverse_index[i];
      if (2*j+1>=x.nr()) {
        cerr << "j out ouf range, this should not happen" << endl;
        continue;
      }
      const double bgr0 = x(2*j+0);
      const double var0 = x(2*j+1);
      const auto [v,der] =
        llg(data[i],bg+bgr0,var+var0,a,universal_distribution);
      value += v;
      without_constraint += v;
      J(2*j+0) += get<0>(der);
      J(2*j+1) += get<1>(der);
      constexpr size_t c = 3;
      if constexpr (c==0)
      for (auto it1 =get<2>(background[i]).begin();
                it1!=get<2>(background[i]).end();
              ++it1) {
        const size_t k = reverse_index[get<0>(*it1)];
        const double w = get<1>(*it1);
        const double bgr1 = x(2*k+0);
        const double var1 = x(2*k+1);
        const double var = abs(var0)+abs(var1);
        double v = 0.5*w*(pow(bgr0-bgr1,2)/var+log(2*pi*var));
        if (isnan(v)||isinf(v)) {
          value += 1e10;
        } else {
          value     += v;
          J(2*j+0)  += w*(bgr0-bgr1)/var;
          J(2*j+1)  += w*0.5*signum(var0)*(1/var-pow((bgr0-bgr1)/var,2));
          J(2*k+0)  += w*(bgr1-bgr0)/var;
          J(2*k+1)  += w*0.5*signum(var1)*(1/var-pow((bgr0-bgr1)/var,2));
        }
      }
      if constexpr (c==1)
      for (auto it1 =get<2>(background[i]).begin();
                it1!=get<2>(background[i]).end();
              ++it1) {
        const size_t k = reverse_index[get<0>(*it1)];
        const double w = pow(get<1>(*it1),2);
        const double bgr1 = x(2*k+0);
        const double var1 = x(2*k+1);
        const double var = abs(var0)+abs(var1);
        double v = 0.5*(pow(bgr0-bgr1,2)/(var/w)+log(2*pi*(var/w)));
        if (isnan(v)||isinf(v)) {
          value += 1e10;
        } else {
          value     += v;
          J(2*j+0)  += (bgr0-bgr1)/(var/w);
          J(2*j+1)  += 0.5*signum(var0)*(1/var-pow(bgr0-bgr1,2)*w/pow(var,2));
          J(2*k+0)  += (bgr1-bgr0)/(var/w);
          J(2*k+1)  += 0.5*signum(var0)*(1/var-pow(bgr0-bgr1,2)*w/pow(var,2));
        }
      }
      if constexpr (c==2)
      for (auto it1 =get<2>(background[i]).begin();
                it1!=get<2>(background[i]).end();
              ++it1) {
        const size_t k = reverse_index[get<0>(*it1)];
        const double w = pow(get<1>(*it1),2);
        const double bgr1 = x(2*k+0);
        const double var1 = x(2*k+1);
        const double var = abs(var0)+abs(var1);
        double v = 0.5*w*(pow(bgr0-bgr1,2)/(var/w)+log(2*pi*(var/w)));
        if (isnan(v)||isinf(v)) {
          value += 1e10;
        } else {
          value     += v;
          J(2*j+0)  += pow(w,2)*(bgr0-bgr1)/var;
          J(2*j+1)  += 0.5*w*signum(var0)*(1/var-pow(bgr0-bgr1,2)*w/pow(var,2));
          J(2*k+0)  += (bgr1-bgr0)/(var/w);
          J(2*k+1)  += 0.5*w*signum(var0)*(1/var-pow(bgr0-bgr1,2)*w/pow(var,2));
        }
      }
      if constexpr (c==3)
      for (auto it1 =get<2>(background[i]).begin();
                it1!=get<2>(background[i]).end();
              ++it1) {
        const size_t k = reverse_index[get<0>(*it1)];
        const double w = get<1>(*it1);
        const double bgr1 = x(2*k+0);
        const double var1 = x(2*k+1);
        const double var = abs(var0)+abs(var1);
        double v =  0.5*w*(pow(bgr0-bgr1,2)/var+log(2*pi*var))
                   +0.5*w*(pow(sqrt(abs(var0))-sqrt(abs(var1)),2)/var
                           +log(2*pi*var));
        if (isnan(v)||isinf(v)) {
          value += 1e10;
        } else {
          value     += v;
          J(2*j+0)  += w*(bgr0-bgr1)/var;
          J(2*j+1)  += w*0.5*signum(var0)*(2/var-pow((bgr0-bgr1)/var,2));
          J(2*j+1)  += signum(var0)*w*0.5*(
                       (sqrt(abs(var0))-sqrt(abs(var1)))/(sqrt(abs(var0))*var)
                       -pow((sqrt(abs(var0))-sqrt(abs(var1)))/var,2));
          J(2*k+0)  += w*(bgr1-bgr0)/var;
          J(2*k+1)  += w*0.5*signum(var1)*(2/var-pow((bgr0-bgr1)/var,2));
          J(2*k+1)  += signum(var1)*w*0.5*(
                       (sqrt(abs(var1))-sqrt(abs(var0)))/(sqrt(abs(var1))*var)
                       -pow((sqrt(abs(var0))-sqrt(abs(var1)))/var,2));
        }
      }
    }
    cerr << without_constraint << " " << value << endl;
    return {value,J};
  }
};

void test_background_target(
    const background_target& target,
    const matrix<double,0,1>& x
    )
{
  const matrix<double,0,1> J = get<1>(target(x));
  const double epsilon = 1e-1;
  for (long int i=0;i!=x.nr();++i) {
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
};

struct background_functor{
  const double& a;
  const double& g;
  const geometry::geometry& geom;
  const struct source& source;
  const double* data;
  vector<tuple<double,double,double>> radial_median;
  patchmap<
    size_t,
    tuple<
      double,
      double,
      vector<
        tuple<
          size_t,
          double
        >
      >
    >
  > background;
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
    for (size_t i=0;i+m<=pixels.size();i+=m) {
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
    ifstream background_file("background");
    if (background_file.good()) {
      size_t i;
      double mu;
      double var;
      while (background_file>>i >> mu >> var) {
        const auto panel = geom.get_panel(i);
        const auto [fs,ss] = panel(i);
        const bool is_old = background.count(i);
        get<0>(background[i]) = mu;
        get<1>(background[i]) = var;
        if (is_old) continue;
        for (int64_t iss = int64_t(ss)-1;iss<=int64_t(ss)+1;++iss) {
          for (int64_t ifs = int64_t(fs)-1;ifs<=int64_t(fs)+1;++ifs) {
            if (!panel.isvalid(ifs,iss)) continue;
            const double j = panel(ifs,iss);
            if (j==i) continue;
            if (background.count(j)==0) continue;
            double x = abs(int64_t(ss)-iss);
            double y = abs(int64_t(fs)-ifs);
            double w = 1 * ( 4.0 + abs(x*y) - 2*(abs(x)+abs(y)) );
            get<2>(background[j]).push_back({i,w});
            get<2>(background[i]).push_back({j,w});
          }
        }
      }
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
      if (it==radial_median.end()) {
        return {data[i],pow(g,2)+pow(data[i],2)};
      }
      if (it!=radial_median.begin())
        if (abs(get<0>(*(it-1))-d)<abs(get<0>(*it)-d)) --it;
      //cerr << d << " " << get<0>(*it) << " "
      //                 << get<1>(*it) << " "
      //                 << get<2>(*it) << endl; 
      return {data[i],pow(g,2)+pow(get<2>(*it),2)};
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
              [&pixels = as_const(pixels)](const auto& entry)
              {return pixels.count(get<0>(entry))==0;}
              ),
            get<2>(it->second).end()
            );
        for (auto it1 =get<2>(it->second).begin();
                  it1!=get<2>(it->second).end();
                ++it1) {
          if (background.count(get<0>(*it1))==0)
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
          double x = abs(int64_t(ss)-iss);
          double y = abs(int64_t(fs)-ifs);
          double w = 1 * ( 4.0 + abs(x*y) - 2*(abs(x)+abs(y)) );
          get<2>(background[j]).push_back({i,w});
          get<2>(background[i]).push_back({j,w});
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
  const void median_mad(
      const double* data,
      const uint8_t* mask,
      const geometry::geometry& geom,
      const vector<tuple<IDX,vector<tuple<
        size_t,double,double,double,struct crystl,struct crystl,struct crystl
      >>>>& prediction
      )
  {
    const double radius = 9;
    patchmap<
      tuple<IDX,const geometry::panel*>,
      tuple<double,double,double>
    > peaks;
    for (const auto & [hkl,pixels] : prediction) {
      for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) { 
        const auto& p = geom.get_panel(i);
        auto& [x,y,w] = peaks[{hkl,&p}];
        w+=flx*wvn;
        const auto [fs,ss] = p(i);
        x+=(fs+0.5)*flx*wvn;
        y+=(ss+0.5)*flx*wvn;
      }
    }
    //cerr << "peaks.size() = " << peaks.size() << endl;
    patchmap<
      tuple<IDX,const geometry::panel*>,
      vector<tuple<double,size_t>>
    > values;
    for (auto it=peaks.begin();it!=peaks.end();++it) {
      const size_t fs = floor(get<0>(it->second)/=get<2>(it->second));
      const size_t ss = floor(get<1>(it->second)/=get<2>(it->second));
      //cerr << fs << " " << ss << endl;
      const auto& panel = *get<1>(it->first);
      for (int64_t iss =int64_t(ss)-ceil(radius);
                   iss<=int64_t(ss)+ceil(radius);
                 ++iss) {
        for (int64_t ifs =int64_t(fs)-ceil(radius);
                     ifs<=int64_t(fs)+ceil(radius);
                   ++ifs) {
          if (pow(int64_t(fs)-ifs,2)+pow(int64_t(ss)-iss,2)>pow(radius,2))
            continue;
          if (!panel.isvalid(ifs,iss)) continue;
          const size_t j = panel(ifs,iss);
          if (mask[j]) continue;
          values[it->first].emplace_back(data[j],j);
        }
      }
    }
    patchmap<
      tuple<IDX,const geometry::panel*>,tuple<double,double>
    > idxg_medianmad;
    //cerr << "values.size() = " << values.size() << endl;
    for (auto it0=values.begin();it0!=values.end();++it0) {
      if (it0->second.size()==0) continue;
      nth_element(it0->second.begin(),
                  it0->second.begin()+it0->second.size()/2,
                  it0->second.end());
      nth_element(it0->second.begin(),
                  it0->second.begin()+(it0->second.size()-1)/2,
                  it0->second.end());
      const double median = 0.5*(
          get<0>(*(it0->second.begin()+ it0->second.size()   /2))
        + get<0>(*(it0->second.begin()+(it0->second.size()-1)/2)));
      for (auto & [v,i] : it0->second) v=abs(v-median);
      nth_element(it0->second.begin(),
                  it0->second.begin()+it0->second.size()/2,
                  it0->second.end());
      nth_element(it0->second.begin(),
                  it0->second.begin()+(it0->second.size()-1)/2,
                  it0->second.end());
      const double mad = 0.5*(
          get<0>(*(it0->second.begin()+ it0->second.size()   /2))
        + get<0>(*(it0->second.begin()+(it0->second.size()-1)/2)));
      idxg_medianmad[it0->first]={median,2.25*pow(mad,2)};
      //cerr << median << " " << mad << " " << it0->second.size() << endl;
    }
    patchmap<size_t,tuple<double,double,double>> compute_background;
    for (auto it=peaks.begin();it!=peaks.end();++it) {
      const size_t fs = floor(get<0>(it->second));
      const size_t ss = floor(get<1>(it->second));
      const auto& panel = *get<1>(it->first);
      for (int64_t iss =int64_t(ss)-ceil(radius);
                   iss<=int64_t(ss)+ceil(radius);
                 ++iss) {
        for (int64_t ifs =int64_t(fs)-ceil(radius);
                     ifs<=int64_t(fs)+ceil(radius);
                   ++ifs) {
          if (pow(int64_t(fs)-ifs,2)+pow(int64_t(ss)-iss,2)>pow(radius,2))
            continue;
          if (!panel.isvalid(ifs,iss)) continue;
          const size_t j = panel(ifs,iss);
          if (mask[j]) continue;
          //cerr << ifs << " " << iss << " " << j << endl;
          get<0>(compute_background[j]) += get<0>(idxg_medianmad[it->first]);
          get<1>(compute_background[j]) += get<1>(idxg_medianmad[it->first]);
          get<2>(compute_background[j]) += 1;
        }
      }
    }
    //cerr << compute_background.size() << endl;
    for (auto it=compute_background.begin();it!=compute_background.end();++it) {
      get<0>(it->second)/=get<2>(it->second);
      get<1>(it->second)/=get<2>(it->second);
      get<0>(background[it->first]) = get<0>(it->second);
      get<1>(background[it->first]) = get<1>(it->second);
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
    j = 0;
    //test_background_target(target,x);
    find_min_gradient(x,target,count_stop_strategy{4096,4096},g*1e-10);
    //find_min_gradient(x,target,count_stop_strategy{1024,1024},g*1e-4);
    //find_min_gradient(x,target,count_stop_strategy{163,163},g*1e-4);
    for (auto it=background.begin();it!=background.end();++it) {
      get<0>(it->second) = x(2*j+0);
      get<1>(it->second) = abs(x(2*j+1));
      ++j;
    }
    ofstream background_file("background");
    for (auto it=background.begin();it!=background.end();++it) {
      background_file << it->first << " "
                      << get<0>(it->second) << " "
                      << get<1>(it->second) << endl;
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
    for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
      if (!pixel_mask.count(i)) continue;
      const auto [mv,mv_nabla] = prediction_mv(g,intensity*flx,wvn,bnd,b,c);
      get<0>(rendering[i])+=mv(0);
      get<1>(rendering[i])+=mv(1);
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
  //const background_functor& background;
  const const_background_functor& background;
  const patchmap<IDX,double>& intensities;
  const vector<source>& sources;
  const double& g;
  const double& a;
  const geometry::geometry geom;
  vector<IDX> indices;
  crystal_target(
      const double* data,
      const patchmap<size_t,void>& pixel_mask,
      //const background_functor& background,
      const const_background_functor& background,
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
    for (auto it=intensities.begin();it!=intensities.end();++it) {
      indices.push_back(it->first);
    }
  }
  const tuple<double,matrix<double,21,1>>
  operator()(const matrix<double,21,1>& x) const {
    //cerr << "in crystal target orperator()" << endl;
    //cerr << trans(x) << endl;
    struct crystl crystl;
    double b,c;
    //cerr << "set_crystal_from_vector" << endl;
    set_crystal_from_vector(x,crystl,b,c);
    //cerr << "predict" << endl;
    auto prediction = predict(sources,crystl,geom,indices,min_flx,oversampling);
    //cerr << "render" << endl;
    const auto rendering =
      render(prediction,g,b,c,intensities,background,pixel_mask);
    matrix<double,3,3> peak = zeros_matrix<double>(3,3);
    peak(0,0) = x(10);
    peak(0,1) = x(11);
    peak(0,2) = x(12);
    peak(1,1) = x(13);
    peak(1,2) = x(14);
    peak(2,2) = x(15);
    //cerr << "rerdering.size() = " << rendering.size() << endl;
    matrix<double,21,1> J = zeros_matrix<double>(21,1);
    for (const auto & [hkl,pixels] : prediction) {
      if (intensities.count(hkl)==0) continue;
      const double intensity = intensities.at(hkl);
      for (const auto & [i,_flx,wvn,bnd,_dflx,_dwvn,_dbnd] : pixels) {
        if (!rendering.count(i)) continue;
        const auto [bg,var_bg] = rendering.at(i);
        const double flx = intensity*_flx;
        const auto [mv,_mv_nabla] = prediction_mv(g,flx,wvn,bnd,b,c);
        matrix<double,3,2> mv_nabla;
        mv_nabla(0,0) = intensity*_mv_nabla(1,0);
        mv_nabla(0,1) = intensity*_mv_nabla(1,1);
        mv_nabla(1,0) = _mv_nabla(2,0);
        mv_nabla(1,1) = _mv_nabla(2,1);
        mv_nabla(2,0) = _mv_nabla(3,0);
        mv_nabla(2,1) = _mv_nabla(3,1);
        struct crystl dflx = _dflx;
        dflx.peak = 2*peak*_dflx.peak;
        struct crystl dwvn = _dwvn;
        dwvn.peak = 2*peak*_dwvn.peak;
        struct crystl dbnd = _dbnd;
        dbnd.peak = 2*peak*_dbnd.peak;
        matrix<double,3,19> prediction_nabla;
        prediction_nabla(0, 0) = dflx.R(0,0);
        prediction_nabla(0, 1) = dflx.R(0,1);
        prediction_nabla(0, 2) = dflx.R(0,2);
        prediction_nabla(0, 3) = dflx.R(1,0);
        prediction_nabla(0, 4) = dflx.R(1,1);
        prediction_nabla(0, 5) = dflx.R(1,2);
        prediction_nabla(0, 6) = dflx.R(2,0);
        prediction_nabla(0, 7) = dflx.R(2,1);
        prediction_nabla(0, 8) = dflx.R(2,2);
        prediction_nabla(0, 9) = dflx.mosaicity;
        prediction_nabla(0,10) = dflx.peak(0,0);
        prediction_nabla(0,11) = dflx.peak(0,1);
        prediction_nabla(0,12) = dflx.peak(0,2);
        prediction_nabla(0,13) = dflx.peak(1,1);
        prediction_nabla(0,14) = dflx.peak(1,2);
        prediction_nabla(0,15) = dflx.peak(2,2);
        prediction_nabla(0,16) = dflx.strain;
        prediction_nabla(0,17) = dflx.a;
        prediction_nabla(0,18) = dflx.b;

        prediction_nabla(1, 0) = dwvn.R(0,0);
        prediction_nabla(1, 1) = dwvn.R(0,1);
        prediction_nabla(1, 2) = dwvn.R(0,2);
        prediction_nabla(1, 3) = dwvn.R(1,0);
        prediction_nabla(1, 4) = dwvn.R(1,1);
        prediction_nabla(1, 5) = dwvn.R(1,2);
        prediction_nabla(1, 6) = dwvn.R(2,0);
        prediction_nabla(1, 7) = dwvn.R(2,1);
        prediction_nabla(1, 8) = dwvn.R(2,2);
        prediction_nabla(1, 9) = dwvn.mosaicity;
        prediction_nabla(1,10) = dwvn.peak(0,0);
        prediction_nabla(1,11) = dwvn.peak(0,1);
        prediction_nabla(1,12) = dwvn.peak(0,2);
        prediction_nabla(1,13) = dwvn.peak(1,1);
        prediction_nabla(1,14) = dwvn.peak(1,2);
        prediction_nabla(1,15) = dwvn.peak(2,2);
        prediction_nabla(1,16) = dwvn.strain;
        prediction_nabla(1,17) = dwvn.a;
        prediction_nabla(1,18) = dwvn.b;

        prediction_nabla(2, 0) = dbnd.R(0,0);
        prediction_nabla(2, 1) = dbnd.R(0,1);
        prediction_nabla(2, 2) = dbnd.R(0,2);
        prediction_nabla(2, 3) = dbnd.R(1,0);
        prediction_nabla(2, 4) = dbnd.R(1,1);
        prediction_nabla(2, 5) = dbnd.R(1,2);
        prediction_nabla(2, 6) = dbnd.R(2,0);
        prediction_nabla(2, 7) = dbnd.R(2,1);
        prediction_nabla(2, 8) = dbnd.R(2,2);
        prediction_nabla(2, 9) = dbnd.mosaicity;
        prediction_nabla(2,10) = dbnd.peak(0,0);
        prediction_nabla(2,11) = dbnd.peak(0,1);
        prediction_nabla(2,12) = dbnd.peak(0,2);
        prediction_nabla(2,13) = dbnd.peak(1,1);
        prediction_nabla(2,14) = dbnd.peak(1,2);
        prediction_nabla(2,15) = dbnd.peak(2,2);
        prediction_nabla(2,16) = dbnd.strain;
        prediction_nabla(2,17) = dbnd.a;
        prediction_nabla(2,18) = dbnd.b;
       
        // 3x19 * 3x2
        // 19x3 * 3x2 = 19x2
        const matrix<double,19,2> dmv_dcrystl =
          trans(prediction_nabla)*mv_nabla;
        matrix<double,2,21> Ji;
        for (size_t i=0;i!=19;++i) {
          Ji(0,i)=dmv_dcrystl(i,0);
          Ji(1,i)=dmv_dcrystl(i,1);
        }
        Ji(0,19) = _mv_nabla(4,0);
        Ji(1,19) = _mv_nabla(4,1);
        Ji(0,20) = _mv_nabla(5,0);
        Ji(1,20) = _mv_nabla(5,1);
        const auto [v,der] =
          llg(data[i],bg,var_bg,a,universal_distribution);
        //cerr << i << " " << data[i] << " " << bg << " " << sqrt(var_bg) << " "
        //     << get<0>(background(i)) << " " << get<1>(background(i)) << endl;
        matrix<double,2,1> dmds{get<0>(der),get<1>(der)};
        //cerr << v << endl;
        //cerr << trans(dmds) << endl;
        //cerr << trans(dmds)*Ji << endl;
        J+=trans(Ji)*dmds;
      }
    }
    //cerr << "in crystal target" << endl;
    //cerr << "x=" << endl;
    //cerr << trans(x) << endl;
    //cerr << "J=" << endl;
    //cerr << trans(J) << endl;
    const double value = common_target(a,data,rendering);
    cerr << value << endl;
    return {value,J};
  }
};

struct crystal_target_cell{
  struct crystal_target target;
  matrix<double,21,1> x0;
  const tuple<double,matrix<double,9,1>>
  operator()(const matrix<double,9,1>& x) const {
    matrix<double,21,1> x1 = x0;
    for (long int i=0;i!=9;++i) x1(i)=x(i);
    const auto [v,_J] = target(x1);
    matrix<double,9,1> J = zeros_matrix<double>(9,1);
    for (long int i=0;i!=9;++i) J(i)=_J(i);
    return {v,J};
  }
};

struct crystal_target_peak{
  struct crystal_target target;
  matrix<double,21,1> x0;
  const tuple<double,matrix<double,6,1>>
  operator()(const matrix<double,6,1>& x) const {
    matrix<double,21,1> x1 = x0;
    for (long int i=0;i!=6;++i) x1(10+i)=x(i);
    const auto [v,_J] = target(x1);
    matrix<double,6,1> J = zeros_matrix<double>(6,1);
    for (long int i=0;i!=6;++i) J(i)=_J(10+i);
    return {v,J};
  }
};

struct individual_foreground_target{
  const double& g;
  const double& a;
  const double& b;
  const double& c;
  const double* data;
  // x flx wvn bnd bg var_bg
  vector<tuple<size_t,double,double,double,double,double>> pixels;
  template<typename T>
  individual_foreground_target(
      const double& g,
      const double& a,
      const double& b,
      const double& c,
      const double* data,
      const T& entry,
      const double intensity,
      const patchmap<size_t,tuple<double,double>>& rendering,
      const IDX& index
      ):g(g),a(a),b(b),c(c),data(data)
  {
    for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : entry) {
      if (rendering.count(i)==0) continue;
      const auto [mv,mv_nabla] =
        prediction_mv(g,intensity*flx,wvn,bnd,b,c);
      const auto& [bg,var_bg] = rendering.at(i);
      pixels.emplace_back(i,flx,wvn,bnd,bg-mv(0),var_bg-mv(1));
    }
  }
  const tuple<double,matrix<double,1,1>>
  operator()(const matrix<double,1,1>& x) const {
    double value = 0;
    matrix<double,1,1> J = zeros_matrix<double>(1,1);
    for (auto it=pixels.begin();it!=pixels.end();++it) {
      const auto& [i,flx,wvn,bnd,bg,var_bg] = *it;
      const auto [mv,mv_nabla] = prediction_mv(g,x(0)*flx,wvn,bnd,b,c);
      const double d = data[i];
      const auto [v,llg_der] =
        llg(d,bg+mv(0),var_bg+mv(1),a,universal_distribution);
      J(0) += flx*x(0)
          *(mv_nabla(1,0)*get<0>(llg_der)
           +mv_nabla(1,1)*get<1>(llg_der));
      value+=v;
    }
    return {value,J};
  }
  void update_rendering(
      const double intensity,
      patchmap<size_t,tuple<double,double>>& rendering
      ) const {
    for (auto it=pixels.begin();it!=pixels.end();++it) {
      const auto& [i,flx,wvn,bnd,bg,var_bg] = *it;
      const auto [mv,mv_nabla] =
        prediction_mv(g,intensity*flx,wvn,bnd,b,c);
      get<0>(rendering[i]) = bg     + mv(0);
      get<1>(rendering[i]) = var_bg + mv(1);
    }
  }
};

struct foreground_target{
  const double* data;
  const double& g;
  const double& a;
  const double& b;
  const double& c;
  const vector<tuple<IDX,vector<tuple<
    size_t,double,double,double,struct crystl,struct crystl,struct crystl
    >>>>& prediction;
  const patchmap<size_t,void>& pixel_mask;
  //const background_functor& background;
  const const_background_functor& background;
  const geometry::geometry& geom;
  patchmap<IDX,double> intensities;
  patchmap<IDX,size_t> reverse_index;
  vector<IDX> indices;
  foreground_target(
    const double* data,
    const double& g,
    const double& a,
    const double& b,
    const double& c,
    const vector<tuple<IDX,vector<tuple<
      size_t,double,double,double,struct crystl,struct crystl,struct crystl
    >>>>& prediction,
    const patchmap<size_t,void>& pixel_mask,
    //const background_functor& background,
    const const_background_functor& background,
    const geometry::geometry& geom,
    patchmap<IDX,double> intensities
      ) :
    data(data),
    g(g),a(a),b(b),c(c),
    prediction(prediction),
    pixel_mask(pixel_mask),
    background(background),
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
        it->second = x(i);
        ++i;
      }
    }
    matrix<double,0,1> J = zeros_matrix<double>(x.nr(),1);
    const auto rendering =
      render(prediction,g,b,c,intensities,background,pixel_mask);
    for (const auto & [hkl,pixels] : prediction) {
      if (intensities.count(hkl)==0) continue;
      const double intensity = intensities.at(hkl);
      for (const auto & [i,flx,wvn,bnd,_dflx,dwvn,dbnd] : pixels) {
        if (!rendering.count(i)) continue;
        const auto [mu,var] = rendering.at(i);
        const auto [mv,mv_nabla] = prediction_mv(g,intensity*flx,wvn,bnd,b,c);
        const double d = data[i];
        const auto [v,llg_der] = llg(d,mu,var,a,universal_distribution);
        if constexpr (false) {
          const double e = 1e-8;
          const auto mvp =
            get<0>(prediction_mv(g,(intensity+e)*flx,wvn,bnd,b,c));
          const auto mvm =
            get<0>(prediction_mv(g,(intensity-e)*flx,wvn,bnd,b,c));
          const double mp = mu -mv(0)+mvp(0);
          const double vp = var-mv(1)+mvp(1);
          const double mm = mu -mv(0)+mvm(0); 
          const double vm = var-mv(1)+mvm(1);
          const double lp =
            get<0>(llg(d,mp,vp,a,universal_distribution));
          const double lm =
            get<0>(llg(d,mm,vm,a,universal_distribution));
          cout << d << " " << mu << " " << sqrt(var) << endl;
          cout << mp << " " << mm << endl;
          cout << lp << " " << lm << endl;
          cout << flx*signum(x(reverse_index.at(hkl)))
                 *(mv_nabla(1,0)*get<0>(llg_der)
                  +mv_nabla(1,1)*get<1>(llg_der))
               << " " << (lp-lm)/(2*e) << endl;
          const double p_o  = universal_distribution(abs(d));
          const auto [p_g,dg]  = gauss(d,mu,var);
          const double p    = a*p_o + (1-a)*p_g;
          cout << p_o << " " << p_g << " " << p << endl;
          cout << endl;
        }
        J(reverse_index.at(hkl)) +=
          flx //*signum(x(reverse_index.at(hkl)))
          *(mv_nabla(1,0)*get<0>(llg_der)
           +mv_nabla(1,1)*get<1>(llg_der));
      }
    }
    const double value = common_target(a,data,rendering); 
    cerr << value << endl;
    return {value,J};
    double constraint = 0;
    for (long int i=0;i!=x.nr();++i) {
      constraint += pow(x(i)-1,2);
      J(i)+=2*(x(i)-1);
    }
    return {value,J};
  }
};

void test_foreground_target(
    const foreground_target& target,
    const matrix<double,0,1>& x
    )
{
  const double e = pow(2,-16);
  const matrix<double,0,1> J = get<1>(target(x));
  matrix<double,0,1> xp = x;
  matrix<double,0,1> xm = x;
  cout << setprecision(16);
  for (long int i=0;i!=x.nr();++i) {
    xp(i) = x(i)+x(i)*e;
    xm(i) = x(i)-x(i)*e;
    const double d = xp(i)-xm(i);
    cout << J(i) << " "
         << (get<0>(target(xp))-get<0>(target(xm)))/d << " "
         << x(i) << " "
         << get<0>(target(xm)) << " "
         << get<0>(target(x))  << " "
         << get<0>(target(xp)) << endl;
    xp(i) = x(i);
    xm(i) = x(i);
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
    const double& radius = 5.5
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

void inline test_pred_der(
    const matrix<double,21,1>& x,
    const vector<struct source>& sources,
    const geometry::geometry& geom
    ) {
  double b=0,c=0;
  struct crystl crystl;
  constexpr bool skip = false;
  patchmap<tuple<IDX,size_t,size_t,int>,tuple<double,double,double>> data;
  //const double epsilon = 2e-9;
  const double epsilon = 1e-8;
  for (size_t j=0;j!=19;++j) {
    matrix<double,21,1> xp = x;
    xp(j)+=epsilon;
    set_crystal_from_vector(xp,crystl,b,c);
    cerr << j << " " << xp(j) << endl;
    cerr << crystl.R << endl;
    cerr << trans(x) << endl;
    auto prediction = predict(sources,crystl,geom,min_flx,oversampling);
    for (const auto & [hkl,pixels] : prediction) {
      for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
        data[{hkl,i,j,1}] = {flx,wvn,bnd};
        /*
        cout << get<0>(hkl) << " "
             << get<1>(hkl) << " "
             << get<2>(hkl) << " "
             << i           << " "
             << j           << " "
             << 1           << " "
             << flx         << endl;
        */
      }
    } 
    matrix<double,21,1> xm = x;
    xm(j)-=epsilon;
    cerr << j << " " << xm(j) << endl;
    set_crystal_from_vector(xm,crystl,b,c);
    cerr << crystl.R << endl;
    prediction = predict(sources,crystl,geom,min_flx,oversampling);
    for (const auto & [hkl,pixels] : prediction) {
      for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) {
        data[{hkl,i,j,-1}] = {flx,wvn,bnd};
        /*
        cout << get<0>(hkl) << " "
             << get<1>(hkl) << " "
             << get<2>(hkl) << " "
             << i           << " "
             << j           << " "
             << -1          << " "
             << flx         << endl;
        */
      }
    }
  }
  set_crystal_from_vector(x,crystl,b,c);
  const auto prediction = predict(sources,crystl,geom,min_flx,oversampling);
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

struct geometry_proposer{
  double nfs,nss;
  double c;
  geometry_proposer(const double a,const double b)
  {
    nfs = a;
    nss = b;
    c = 12.0/(
             pow(a,2)/b*asinh(1.0*b/a)
             +2*sqrt(pow(a,2)+pow(b,2))
             +pow(b,2)/a*asinh(1.0*a/b)
             );
  }
  std::mt19937 gen;
  std::normal_distribution<double> d{0,1.0/16};
  const inline matrix<double,6,1> operator()(matrix<double,6,1> x) {
    //cerr << trans(x);
    const matrix<double,2,2> M{1+c*d(gen),c*d(gen),c*d(gen),1+c*d(gen)};
    matrix<double,2,2> D{x(0),x(1),x(3),x(4)};
    matrix<double,2,1> o{nfs/2,nss/2};
    o = D*(o-M*o); // zero shift in center of panel
    D = D*M;
    //o(0)+=d(gen); // now shift center
    //o(1)+=d(gen);
    x(0) =D(0,0);
    x(1) =D(0,1);
    x(2)+=o(  0);
    x(3) =D(1,0);
    x(4) =D(1,1);
    x(5)+=o(  1);
    //cerr << trans(x);
    return x;
  }
};

struct geometry_target{
  const double* data;
  //const background_functor& background;
  patchmap<size_t,void> pixel_mask;
  const const_background_functor& background;
  const patchmap<IDX,double>& intensities;
  const vector<source>& sources;
  const struct crystl& crystl;
  const double& g;
  const double& a;
  const double& b;
  const double& c;
  const geometry::geometry& geom;
  vector<IDX> indices;
  geometry_target(
      const double* data,
      const patchmap<size_t,void>& full_pixel_mask,
      //const background_functor& background,
      const const_background_functor& background,
      const patchmap<IDX,double>& intensities,
      const vector<source>& sources,
      const struct crystl& crystl,
      const double& g,
      const double& a,
      const double& b,
      const double& c,
      const geometry::geometry& geom
    ) :
    data(data),
    pixel_mask(full_pixel_mask),
    background(background),
    intensities(intensities),
    sources(sources),
    crystl(crystl),
    g(g),
    a(a),
    b(b),
    c(c),
    geom(geom)
  {
    for (auto it=intensities.begin();it!=intensities.end();++it) {
      indices.push_back(it->first);
    }
    for (auto it=pixel_mask.begin();it!=pixel_mask.end();) {
      if (geom.get_panel(*it).isvalid(*it)) ++it;
      else it=pixel_mask.erase(it);
    }
  }
  const double operator()(const matrix<double,6,1>& x) const {
    auto test = geom;
    test.panels.front().D(0,0) = x(0);
    test.panels.front().D(0,1) = x(1);
    test.panels.front().o(0)   = x(2);
    test.panels.front().D(1,0) = x(3);
    test.panels.front().D(1,1) = x(4);
    test.panels.front().o(1)   = x(5);
    const auto rendering =
      render
        (
          predict(sources,crystl,test,indices,min_flx,oversampling),
          g,b,c,intensities,background,pixel_mask
        );
    const double value = common_target(a,data,rendering);
    //const auto D = test.panels.front().D;
    //const auto o = test.panels.front().o;
    //cerr << D(0,0) << " " << D(0,1) << " " << o(0) << endl;
    //cerr << D(1,0) << " " << D(1,1) << " " << o(1) << endl;
    //cerr << D(2,0) << " " << D(2,1) << " " << o(2) << endl;
    cerr << value << endl;
    return value;
  }
};


template<class F>
void optimise_geometry
(
    const double* data,
    geometry::geometry& geom,
    const patchmap<size_t,void>& pixel_mask,
    const vector<tuple<IDX,vector<tuple<
      size_t,double,double,double,struct crystl,struct crystl,struct crystl
    >>>>& prediction,
    const F& background,
    const double radius = 4
)
{
  patchmap<const geometry::panel*,tuple<double,double,double>> cp;
  patchmap<const geometry::panel*,tuple<double,double,double>> cq;
  patchmap<tuple<IDX,const geometry::panel*>,tuple<double,double,double>>
    predicted_peaks;
  patchmap<tuple<IDX,const geometry::panel*>,tuple<double,double,double>>
    measured_peaks;
  patchmap<const geometry::panel*,size_t> num;
  patchmap<const geometry::panel*,matrix<double,2,2>> pqt;
  patchmap<const geometry::panel*,matrix<double,2,2>> qpt;
  patchmap<const geometry::panel*,matrix<double,2,2>> qqt;
  patchmap<const geometry::panel*,matrix<double,2,2>> ppt;
  double error = 0;
  for (const auto & [hkl,pixels] : prediction) {
    for (const auto & [i,flx,wvn,bnd,dflx,dwvn,dbnd] : pixels) { 
      const auto p = &geom.get_panel(i);
      auto& [x,y,w] = predicted_peaks[{hkl,p}];
      if (pixel_mask.count(i)==0) continue;
      w+=flx*wvn;
      const auto [fs,ss] = (*p)(i);
      x+=(fs+0.5)*flx*wvn;
      y+=(ss+0.5)*flx*wvn;
    }
  }
  for (auto it=predicted_peaks.begin();it!=predicted_peaks.end();++it) {
    const size_t dfs = (get<0>(it->second)/=get<2>(it->second));
    const size_t dss = (get<1>(it->second)/=get<2>(it->second));
    const size_t fs = floor(dfs);
    const size_t ss = floor(dss);
    double x = 0;
    double y = 0;
    double w = 0;
    const auto p = get<1>(it->first);
    for (int64_t iss =int64_t(ss)-ceil(radius);
                 iss<=int64_t(ss)+ceil(radius);
               ++iss) {
      for (int64_t ifs =int64_t(fs)-ceil(radius);
                   ifs<=int64_t(fs)+ceil(radius);
                 ++ifs) {
        const double d = pow(dfs-(ifs+0.5),2)+pow(dss-(iss+0.5),2);
        if (d>=pow(radius,2)) continue;
        if (!(p->isvalid(ifs,iss))) continue;
        const size_t j = (*p)(ifs,iss);
        if (pixel_mask.count(j)==0) continue;
        //if (data[j]<get<0>(background(j))) continue;
        const double t = (data[j]-get<0>(background(j)))
                         *sqrt(1-d/pow(radius,2));
        if (t<0) continue;
        x+=t*(ifs+0.5);
        y+=t*(iss+0.5);
        w+=t;
      }
    }
    if ((w>0)&&(get<2>(it->second)>0)) {
      ++num[get<1>(it->first)];
      x/=w;
      y/=w;
      measured_peaks[it->first] = {x,y,w};
      w *= get<2>(it->second);
      w = sqrt(w);
      get<0>(cp[p])+=w*get<0>(it->second);
      get<1>(cp[p])+=w*get<1>(it->second);
      get<2>(cp[p])+=w;
      get<0>(cq[p])+=w*x;
      get<1>(cq[p])+=w*y;
      get<2>(cq[p])+=w;
      error += pow(x-get<0>(it->second),2);
      error += pow(y-get<1>(it->second),2);
    }
  }
  cerr << "error = " << error << endl;
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    cerr << "o" << endl;
    get<0>(cp[&(*it)])/=get<2>(cp[&(*it)]);
    get<1>(cp[&(*it)])/=get<2>(cp[&(*it)]);
    get<0>(cq[&(*it)])/=get<2>(cq[&(*it)]);
    get<1>(cq[&(*it)])/=get<2>(cq[&(*it)]);
    const double dx = get<0>(cp[&(*it)])-get<0>(cq[&(*it)]);
    const double dy = get<1>(cp[&(*it)])-get<1>(cq[&(*it)]);
    if (num[&(*it)]) it->o += it->D*matrix<double,2,1>{dx,dy};
    cerr << trans(it->o) << endl;
    ppt[&(*it)] = zeros_matrix<double>(2,2);
    pqt[&(*it)] = zeros_matrix<double>(2,2);
    qpt[&(*it)] = zeros_matrix<double>(2,2);
    qqt[&(*it)] = zeros_matrix<double>(2,2);
  }
  for (auto it=predicted_peaks.begin();it!=predicted_peaks.end();++it) {
    if (measured_peaks.count(it->first)==0) continue;
    const double w0 = get<2>(it->second);
    if (w0<=0) continue;
    const double w1 = get<2>(measured_peaks[it->first]);
    if (w1<=0) continue;
    const double w = sqrt(w0*w1);
    matrix<double,2,1> p{get<0>(it->second),get<1>(it->second)};
    p(0)-=get<0>(cp[get<1>(it->first)]);
    p(1)-=get<1>(cp[get<1>(it->first)]);
    matrix<double,2,1> q
      {get<0>(measured_peaks[it->first]),get<1>(measured_peaks[it->first])};
    q(0)-=get<0>(cq[get<1>(it->first)]);
    q(1)-=get<1>(cq[get<1>(it->first)]);
    qqt[get<1>(it->first)]+=w*q*trans(q);
    pqt[get<1>(it->first)]+=w*p*trans(q);
    ppt[get<1>(it->first)]+=w*p*trans(p);
    qpt[get<1>(it->first)]+=w*q*trans(p);
  }
  for (auto it=predicted_peaks.begin();it!=predicted_peaks.end();++it) {
    if (measured_peaks.count(it->first)==0) continue;
    const double w0 = get<2>(it->second);
    if (w0<=0) continue;
    const double w1 = get<2>(measured_peaks[it->first]);
    if (w1<=0) continue;
    const double w = sqrt(w0*w1);
    matrix<double,2,1> p{get<0>(it->second),get<1>(it->second)};
    p(0)-=get<0>(cp[get<1>(it->first)]);
    p(1)-=get<1>(cp[get<1>(it->first)]);
    matrix<double,2,1> q
      {get<0>(measured_peaks[it->first]),get<1>(measured_peaks[it->first])};
    q(0)-=get<0>(cq[get<1>(it->first)]);
    q(1)-=get<1>(cq[get<1>(it->first)]);
    //const matrix<double,2,2> R = qqt[get<1>(it->first)]
    //                        *inv(pqt[get<1>(it->first)]);
    const matrix<double,2,2> R = pqt[get<1>(it->first)]
                            *inv(qqt[get<1>(it->first)]);
    //cerr << trans(p); 
    //cerr << trans(q); 
    //cerr << trans(R*q);
    //cerr << endl;
  }
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    if (num.count(&(*it))==0) continue;
    if (num.at(&(*it))<4) continue;
    const matrix<double,2,2> R = pqt[&(*it)]*inv(qqt[&(*it)]);
    //cerr << "R=" << endl;
    //cerr << R << endl;
    //cerr << it->D;
    //cerr << endl;
    it->D=it->D*R;
    //cerr << endl;
    //cerr << it->D;
    //cerr << endl;
  }
}

template<class F>
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
    const F& background
    ) {
  vector<IDX> indices;
  for (auto it=intensities.begin();it!=intensities.end();++it)
    indices.push_back(it->first);
  auto greed = geom;
  for (auto it=greed.panels.begin();it!=greed.panels.end();++it) {
    geometry::geometry geom;
    geom.num_pixels = greed.num_pixels;
    geom.panels.push_back(*it);
    geometry_target
      target(data,pixel_mask,background,intensities,sources,crystl,g,a,b,c,geom);
    matrix<double,6,1> x;
    x(0) = it->D(0,0);
    x(1) = it->D(0,1);
    x(2) = it->o(  0);
    x(3) = it->D(1,0);
    x(4) = it->D(1,1);
    x(5) = it->o(  1);
    geometry_proposer propose(it->nfs,it->nss);
    find_min_numerical(
        x,
        target,
        propose,
        count_stop_strategy{64,64}
        );
    it->D(0,0) = x(0);
    it->D(0,1) = x(1);
    it->o(  0) = x(2); 
    it->D(1,0) = x(3);
    it->D(1,1) = x(4);
    it->o(  1) = x(5);
  }
  return greed;
  /*
  geometry::geometry best_geom = geom;
  auto prediction = predict(sources,crystl,geom,indices,min_flx,oversampling);
  geometry::geometry greed = geom;
  double best_value =
    common_target(a,data,
        render(prediction,g,b,c,intensities,background,pixel_mask)
        );
  if (true) {
  for (size_t i=0;i!=16;++i) {
    optimise_geometry(data,greed,pixel_mask,prediction,background,5);
    prediction = predict(sources,crystl,greed,indices,min_flx,oversampling);
  }
  double greed_value =
    common_target(a,data,
        render(prediction,g,b,c,intensities,background,pixel_mask)
        );
  cerr << "greedy optimisation " << endl
       << best_value << " " << greed_value << endl;
  //return greed;
  if (greed_value<best_value) {
    best_value = greed_value;
    best_geom = greed;
  }
  }
  random_device rd;
  normal_distribution<double> nd(0,1);
  for (size_t i=0;i!=geom.panels.size();++i) {
    for (size_t j=0;j!=16;++j) {
      greed.panels[i].o(1)  =best_geom.panels[i].o(0)  +nd(rd);
      greed.panels[i].o(1)  =best_geom.panels[i].o(1)  +nd(rd);
      greed.panels[i].D(0,0)=best_geom.panels[i].D(0,0)+nd(rd)/1000;
      greed.panels[i].D(0,1)=best_geom.panels[i].D(0,1)+nd(rd)/1000;
      greed.panels[i].D(1,0)=best_geom.panels[i].D(1,0)+nd(rd)/1000;
      greed.panels[i].D(1,1)=best_geom.panels[i].D(1,1)+nd(rd)/1000;
      const double value =
          common_target(
              a,data,
              render(
                predict(
                  sources,crystl,greed,indices,min_flx,oversampling
                )
                ,g,b,c,intensities,background,pixel_mask
              )
          );
      if (value<best_value) {
        best_geom = greed;
        best_value = value;
        cerr << i << " " << value << endl;
      } else {
        greed = best_geom;
      }
    }
  }
  return best_geom;
  */
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

template<class F>
void save_rendering(
    const patchmap<size_t,tuple<double,double>>& rendering,
    const F& background,
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
          const double e = get<0>(background(i));
          const double v = data[i];
          //const uint8_t n =
          //  clip(4*((mask[i]==0)*log2(1+pow(size_t(v>e?v-e:0),2))),0,255);
          const uint8_t n =
            clip(2*((mask[i]==0)*sqrt(v>e?v-e:0)),0,255);
          file.put(n);
    }
  }
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    ofstream file("panel_"+to_string(it-geom.panels.begin())+".pgm");
    file << "P5 " << it->nfs << " " << it->nss << " " << 255 << endl;
    for (size_t ss=0;ss!=it->nss;++ss) for (size_t fs=0;fs!=it->nfs;++fs) {
          const size_t i = (*it)(fs,ss);
          const double e = get<0>(background(i));
          const double v = rendering.count(i)?get<0>(rendering[i]):e;
          //const uint8_t n =
          //  clip(4*(mask[i]==0)*log2(1+pow(size_t(v>e?v-e:0),2)),0,255);
          const uint8_t n =
            clip(2*((mask[i]==0)*sqrt(v>e?v-e:0)),0,255);
          file.put(n);
    }
  }
}

int main() {
  const double a = pow(1.0/256,2u);
  const auto g = 9511.11*0.00105;
  double b = 1.0;
  double c = 1.0/256;
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
  cerr << "read geometry description" << endl;
  cerr << geom.panels.size() << " panels" << endl;
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
  cerr << "data[196] = " << data[196] << endl;
  double* background_precomputed = new double[geom.num_pixels*sizeof(double)];
  cin.read(reinterpret_cast<char*>(background_precomputed),
           geom.num_pixels*sizeof(double));
  if (!cin) {
    cerr << "ran out of buffer for background" << endl; 
  }
  double* variance_precomputed = new double[geom.num_pixels*sizeof(double)];
  cin.read(reinterpret_cast<char*>(variance_precomputed),
           geom.num_pixels*sizeof(double));
  if (!cin) {
    cerr << "ran out of buffer for variance" << endl; 
  }
  auto crystl = crystls.front();
  auto source = average_source(sources);
  patchmap<IDX,double> intensities;
  //background_functor background(a,g,geom,source,data,mask);
  const_background_functor background
      {background_precomputed,variance_precomputed};
  //return 0;
  // loop:
  //   fit intensities
  //   fit background (or not)
  //   fit crystal
  // output h,k,l,i,δ²,background,δ²_background
  if constexpr (true) {
    ifstream intensityfile("intensities");
    int h,k,l;
    double i;
    while (intensityfile >> h >> k >> l >> i) intensities[{h,k,l}]=i;
  }
  ofstream panel_file("panel_16__oldmodel.pgm");
  const auto& panel = geom.panels[16];
  panel_file << "P5 " << panel.nfs << " " << panel.nss << " " << 255 << endl;
  const double r = 1e-2;
  for (size_t ss=0;ss!=panel.nss;++ss) {
    for (size_t fs=0;fs!=panel.nfs;++fs) {
      const size_t i = panel(fs,ss);
      const double e = get<0>(background(i));
      double v = e;
      const size_t o = 16;
      for (size_t oss=0;oss!=o;++oss) {
        for (size_t ofs=0;ofs!=o;++ofs) {
          matrix<double,2,1> fsss{fs+(ofs+0.5)/o,ss+(oss+0.5)/o};
          matrix<double,3,1> x = panel(fsss);
          x = length(source.kin)*normalize(x)-source.kin;
          matrix<double,3,1> dhkl = crystl.U*x;
          dhkl = round(dhkl);
          IDX hkl{int32_t(dhkl(0)),int32_t(dhkl(1)),int32_t(dhkl(2))};
          const double l = length(crystl.R*dhkl-x);
          if (l<r) v += 0.2*intensities[hkl]/(pow(o,2)*pow(r,3)*1.25*pi);
        }
      }
      const uint8_t n =
        clip(2*((mask[i]==0)*sqrt(v>0?v:0)),0,255);
      panel_file.put(n);
    }
  }
}
