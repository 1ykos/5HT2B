#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include "partiality.hpp"
#include "geometry.hpp"
#include "wmath.hpp"
#include "wmath_optimisation.hpp"

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
using wmath::clip;
using wmath::count_stop_strategy;
using wmath::mean_variance;
using wmath::signum;
using wmath::tetration;
using wmath::universal_distribution;
using wmath::destructive_median_mad;

using geometry::panel;

constexpr double pi = 3.141592653589793;

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

template<class rng>
double generate_universal_real_number(
    rng& gen
    ) {
  uniform_real_distribution<double> distr(0,3.6322001994373);
  return tetration(2.718281828459045,distr(gen))-1;
}

template<class maskclass,class function>
double background_target(
    const double a,
    const function o,
    const double dist,
    const panel* p,
    const maskclass& mask,
    const double* image,
    const double* background,
    const double* variance,
          double* dbackground,
          double* dvariance
    ){
  fill(dbackground,dbackground+p->nfs*p->nss,0.0);
  fill(dvariance  ,dvariance  +p->nfs*p->nss,0.0);
  double value = 0;
  for (size_t ss=0;ss!=p->nss;++ss) {
    for (size_t fs=0;fs!=p->nfs;++fs) {
      const size_t i = ss*p->nfs+fs;
      if (!mask[i]) {
        const auto tmp =
          llg(image[i],background[i],variance[i],a,o);
        value += get<0>(tmp);
        dbackground[i] += get<0>(get<1>(tmp));
          dvariance[i] += get<1>(get<1>(tmp));
      }
      if ((fs-1<p->nfs)&&(fs+1<p->nfs)) {
        const double mean_val = 0.5*(background[i-     1]+background[i+     1]);
        const double mean_var = 0.5*(  variance[i-     1]+  variance[i+     1]);
        // smoothness of variance
        const double var_ratio = variance[i]/mean_var;
        value+=dist*(log(sqrt(pi)/2)-2+pow(var_ratio,2)+pow(var_ratio,-2));
        const double dr = dist*2*(var_ratio-pow(var_ratio,-3));
        dvariance  [i]+= dr/mean_var;
        const double drn=
          dist*(mean_var/pow(variance[i],2)-pow(var_ratio,2)/mean_var);
        dvariance  [i-  1]+=drn;
        dvariance  [i+  1]+=drn;
        // smoothness of background
        const double l = 0.5*dist*pow(mean_val-background[i],2)
                            /(mean_var+variance[i]);
        value += l;
        dbackground[i]+=dist*(background[i]-mean_val)/(mean_var+variance[i]);
        dvariance  [i]+=
          -0.5*dist*pow((mean_val-background[i])/(mean_var+variance[i]),2);
        const double dmean_val_n = 0.5*dist*(mean_val-background[i])
                                           /(mean_var+variance[i]); 
        const double dmean_var_n =
          -0.25*dist*pow((mean_val-background[i])/(mean_var+variance[i]),2);
        dbackground[i-     1]+=dmean_val_n;
        dbackground[i+     1]+=dmean_val_n;
        dvariance  [i-     1]+=dmean_var_n;
        dvariance  [i+     1]+=dmean_var_n;
      }
      if ((ss-1<p->nss)&&(ss+1<p->nss)) {
        const double mean_val = 0.5*(background[i-p->nfs]+background[i+p->nfs]);
        const double mean_var = 0.5*(  variance[i-p->nfs]+  variance[i+p->nfs]);
        // smoothness of variance
        const double var_ratio = variance[i]/mean_var;
        value+=dist*(log(sqrt(pi)/2)-2+pow(var_ratio,2)+pow(var_ratio,-2));
        const double dr = dist*2*(var_ratio-pow(var_ratio,-3));
        dvariance  [i]+= dr/mean_var;
        const double drn=
          dist*(mean_var/pow(variance[i],2)-pow(var_ratio,2)/mean_var);
        dvariance  [i-p->nfs]+=drn;
        dvariance  [i+p->nfs]+=drn;
        // smoothness of background
        const double l = 0.5*dist*pow(mean_val-background[i],2)
                            /(mean_var+variance[i]);
        value += l;
        dbackground[i]+=dist*(background[i]-mean_val)/(mean_var+variance[i]);
        dvariance  [i]+=
          -0.5*dist*pow((mean_val-background[i])/(mean_var+variance[i]),2);
        const double dmean_val_n = 0.5*dist*(mean_val-background[i])
                                           /(mean_var+variance[i]); 
        const double dmean_var_n =
          -0.25*dist*pow((mean_val-background[i])/(mean_var+variance[i]),2);
        dbackground[i-p->nfs]+=dmean_val_n;
        dbackground[i+p->nfs]+=dmean_val_n;
        dvariance  [i-p->nfs]+=dmean_var_n;
        dvariance  [i+p->nfs]+=dmean_var_n;
      }
    }
  }
  return value;
}

template<class maskclass,class function>
void test_derivative(
    const double a,
    const function o,
    const double dist,
    const panel* p,
    const maskclass& mask,
    const double* image,
          double* background,
          double* variance
    ) {
  double eps = 1;
  double* dbackground = new double[p->nfs*p->nss];
  double* dvariance   = new double[p->nfs*p->nss];
  for (size_t i=0;i!=p->nfs*p->nss;++i) {
    const double b = background[i];
    background[i]=b+eps;
    const double pb = background_target
      (a,o,dist,p,mask,image,background,variance,dbackground,dvariance);
    background[i]=b-eps;
    const double mb = background_target
      (a,o,dist,p,mask,image,background,variance,dbackground,dvariance);
    background[i]=b;
    const double v = variance[i];
    variance[i]=v+eps;
    const double pv = background_target
      (a,o,dist,p,mask,image,background,variance,dbackground,dvariance);
    variance[i]=v-eps;
    const double mv = background_target
      (a,o,dist,p,mask,image,background,variance,dbackground,dvariance);
    variance[i]=v;
    cout << dbackground[i] << " " << (pb-mb)/(2*eps) << " "
         << dvariance[i] << " " << (pv-mv)/(2*eps) << " "
         << (mask[i]?1:0) << endl;
  }
  delete[] dbackground;
  delete[] dvariance;
}

template<class maskclass,class function>
void optimise_background(
    const double a,
    const function o,
    const double dist,
    const panel* p,
    const maskclass& mask,
    const double* image,
    double *background,
    double *variance
    ) {
  const double  phi = (sqrt(5.0) + 1.0) * 0.5;
  const double iphi = 1.0/phi;
  double beta0 = 1;
  double beta1 = 1;
  double* dbackground = new double[p->nfs*p->nss];
  double* dvariance   = new double[p->nfs*p->nss];
  double* background_ = new double[p->nfs*p->nss];
  double* variance_   = new double[p->nfs*p->nss];
  double* dump0       = new double[p->nfs*p->nss];
  double* dump1       = new double[p->nfs*p->nss];
  double norm,c;
  double value = background_target
      (a,o,dist,p,mask,image,background,variance,dbackground,dvariance);
  for (size_t i=0;i!=(1u<<8);++i) {
    //cerr << '\r' << setw(8) << i;
    cerr << setw(8) << i << " " << value << endl;
    norm = sqrt(accumulate(dbackground,dbackground+p->nfs*p->nss,0.0,
        [](const auto l,const auto r){return l+r*r;}));
    while(beta0>1e-32) {
      const double c = beta0*p->nfs*p->nss/norm;
      transform(background,background+p->nfs*p->nss,dbackground,background_,
          [&c](const auto v,const auto d){return v-c*d;});
      const double test = background_target
        (a,o,dist,p,mask,image,background_,variance,dump0,dump1);
      if (test<value) {
        value = test;
        beta0*=2;
        copy(background_,background_+p->nfs*p->nss, background);
        copy(dump0      ,dump0      +p->nfs*p->nss,dbackground);
        copy(dump1      ,dump1      +p->nfs*p->nss,dvariance  );
        break;
      }
      beta0*=iphi;
    }
    norm = sqrt(accumulate(dvariance  ,dvariance  +p->nfs*p->nss,0.0,
        [](const auto l,const auto r){return l+r*r;}));
    c = beta1*p->nfs*p->nss/norm;
    while(beta1>1e-32) {
      const double c = beta1*p->nfs*p->nss/norm;
      transform(variance  ,variance  +p->nfs*p->nss,dvariance  ,  variance_,
          [&c](const auto v,const auto d){return abs(v-c*d);});
      const double test = background_target
        (a,o,dist,p,mask,image,background,variance_,dump0,dump1);
      if (test<value) {
        value = test;
        beta1*=2;
        copy(  variance_,  variance_+p->nfs*p->nss, variance  );
        copy(dump0      ,dump0      +p->nfs*p->nss,dbackground);
        copy(dump1      ,dump1      +p->nfs*p->nss,dvariance  );
        break;
      }
      beta1*=iphi;
    }
    if ((beta0<1e-32)&&(beta1<1e-32)) break;
  }
  cerr << endl;
  delete[] dbackground;
  delete[] dvariance;
  delete[] background_;
  delete[] variance_;
  delete[] dump0;
  delete[] dump1;
}

template<class maskclass,class function>
void optimise_background2(
    const double a,
    const function o,
    const double dist,
    const panel* p,
    const maskclass& mask,
    const double* image,
    double *background,
    double *variance
    ) {
  double beta = 1.0/128;
  double* dbackground = new double[p->nfs*p->nss];
  double* dvariance   = new double[p->nfs*p->nss];
  for (size_t i=0;i!=(1u<<8);++i) {
    //cerr << '\r' << setw(8) << i;
    const double value = background_target
      (a,o,dist,p,mask,image,background,variance,dbackground,dvariance);
    cerr << setw(8) << i << " " << value << endl;
    double norm_dbackground = 0;
    for (size_t i=0;i!=p->nfs*p->nss;++i) norm_dbackground+=pow(dbackground[i],2);
    norm_dbackground = beta/sqrt(norm_dbackground)*(p->nfs*p->nss);
    double norm_dvariance = 0;
    for (size_t i=0;i!=p->nfs*p->nss;++i) norm_dvariance+=pow(dvariance[i],2);
    norm_dvariance = beta/sqrt(norm_dvariance)*(p->nfs*p->nss);
    for (size_t i=0;i!=p->nfs*p->nss;++i)
      background[i]-=norm_dbackground*dbackground[i];
    for (size_t i=0;i!=p->nfs*p->nss;++i)
      variance[i]=abs(variance[i]-norm_dvariance*dvariance[i]);
  }
  delete[] dbackground;
  delete[] dvariance;
}

int main() {
  const double a = 1.0/256;
  const auto g = 9511.11*0.00105;
  geometry::geometry geom;
  if (!read_geometry_bin(cin,geom)) {
    cerr << "read error in binary geometry file" << endl;
    return 3;
  }
  cerr << "read geometry description "
       << geom.panels.size()
       << " panels" << endl;
  uint8_t* mask = new uint8_t[geom.num_pixels];
  // read in mask data
  cin.read(reinterpret_cast<char*>(mask),geom.num_pixels*sizeof(uint8_t));
  if (!cin) {
    cerr << "ran out of buffer for mask" << endl;
    return 5;
  }
  cerr << "read mask" << endl;
  double* data = new double[geom.num_pixels*sizeof(double)];
  // read in image data
  cin.read(reinterpret_cast<char*>(data),geom.num_pixels*sizeof(double));
  if (!cin) {
    cerr << "ran out of buffer for data" << endl;
    return 6;
  }
  cerr << "read image data" << endl;
  double* background = new double[geom.num_pixels*sizeof(double)];
  double* variance   = new double[geom.num_pixels*sizeof(double)];
  copy(data,data+geom.num_pixels,background);
  ifstream background_variance("background_variance");
  if (background_variance) {
    background_variance.read(reinterpret_cast<char*>(background),
                             geom.num_pixels*sizeof(double));
    background_variance.read(reinterpret_cast<char*>(variance),
                             geom.num_pixels*sizeof(double));
  }
  cerr << setprecision(16);
  for (auto it=geom.panels.begin();it!=geom.panels.end();++it) {
    if (false) {
    const auto median_mad = destructive_median_mad
      (background+it->d,background+it->d+it->nfs*it->nss);
    fill(background+it->d,background+it->d+it->nfs*it->nss,get<0>(median_mad));
    fill(variance  +it->d,variance  +it->d+it->nfs*it->nss,get<1>(median_mad)+g);
    }
    optimise_background2(
        a,
        universal_distribution,
        32,
        &(*it),
        mask,
        data+it->d,
        background+it->d,
        variance+it->d
        );
    optimise_background(
        a,
        universal_distribution,
        32,
        &(*it),
        mask,
        data+it->d,
        background+it->d,
        variance+it->d
        );
    {
    ofstream file("panel_"+to_string(it-geom.panels.begin())+"_background.pgm");
    file << "P5 " << it->nfs << " " << it->nss << " " << 255 << endl;
    for (size_t ss=0;ss!=it->nss;++ss) for (size_t fs=0;fs!=it->nfs;++fs) {
          const size_t i = (*it)(fs,ss);
          const double v = background[i];
          const uint8_t n =
            clip(4*(mask[i]==0)*log2(1+pow(size_t(v),2)),0,255);
          file.put(n);
    }
    }
    {
    ofstream file("panel_"+to_string(it-geom.panels.begin())+"_variance.pgm");
    file << "P5 " << it->nfs << " " << it->nss << " " << 255 << endl;
    for (size_t ss=0;ss!=it->nss;++ss) for (size_t fs=0;fs!=it->nfs;++fs) {
          const size_t i = (*it)(fs,ss);
          const double v = variance[i];
          const uint8_t n =
            clip(4*(mask[i]==0)*log2(1+pow(size_t(v),2)),0,255);
          file.put(n);
    }
    }
    {
    ofstream file("background_variance");
    file.write(reinterpret_cast<char*>(background),
               geom.num_pixels*sizeof(double));
    file.write(reinterpret_cast<char*>(variance),
               geom.num_pixels*sizeof(double));
    }
  }
}
