#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include "partiality.hpp"
#include "geometry.hpp"
#include "wmath.hpp"
#include "wmath_optimisation.hpp"

using dlib::abs;
using dlib::matrix;
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
using std::uniform_real_distribution;
using std::bernoulli_distribution;
using std::poisson_distribution;
using std::normal_distribution;
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
using wmath::clip;
using wmath::count_stop_strategy;
using wmath::mean_variance;
using wmath::signum;
using wmath::tetration;
using wmath::universal_distribution;

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

template<class rng>
tuple<double,double> true_background(
    double x,
    double y,
    rng& gen,
    const double a,
    const double g
    ) {
  bernoulli_distribution bd(a);
  double n = 5*exp(-0.5*(pow(x,  2)+pow(y,2))/pow(70,2));
  double v = n;
  if (bd(gen)) {
    v = generate_universal_real_number(gen);
    return {g*n,v};
  }
  v += 20*exp(-0.5*(pow(x-7*7,2)+pow(y-7*5,2)));
  v += 20*exp(-0.5*(pow(x+7*3,2)+pow(y+7*5,2)));
  v += 20*exp(-0.5*(pow(x+3*3,2)+pow(y+4*5,2)));
  v += 20*exp(-0.5*(pow(x-7*8,2)+pow(y-7*2,2)));
  v += 20*exp(-0.5*(pow(x-7*3,2)+pow(y+7*4,2)));
  v += 20*exp(-0.5*(pow(x    ,2)+pow(y-7*6,2)));
  return {g*n,normal_distribution<double>(0,g)(gen)
             +g*poisson_distribution<size_t>(v)(gen)};
}

double background_target(
    const size_t nfs,
    const size_t nss,
    const double a,
    const double dist,
    const double* image,
    const double* background,
    const double* variance,
          double* dbackground,
          double* dvariance
    ){
  for (size_t j=0;j!=nfs*nss;++j) dbackground[j]=0;
  for (size_t j=0;j!=nfs*nss;++j) dvariance  [j]=0;
  double value = 0;
  for (size_t ss=0;ss!=nss;++ss) {
    for (size_t fs=0;fs!=nfs;++fs) {
      const size_t i = ss*nfs+fs;
      if (true) { // is not masked, there is no mask here
        const auto tmp =
          llg(image[i],background[i],variance[i],a,universal_distribution);
        value += get<0>(tmp);
        dbackground[ss*nfs+fs] += get<0>(get<1>(tmp));
          dvariance[ss*nfs+fs] += get<1>(get<1>(tmp));
      }
      if ((fs-1<nfs)&&(fs+1<nfs)) {
        const double mean_val =
          0.5*(background[(ss+0)*nfs+(fs-1)]+background[(ss+0)*nfs+(fs+1)]);
        const double mean_var =
          0.5*(  variance[(ss+0)*nfs+(fs-1)]+  variance[(ss+0)*nfs+(fs+1)]);
        // smoothness of variance
        const double var_ratio = variance[i]/mean_var;
        value+=dist*(log(sqrt(pi)/2)-(2-pow(var_ratio,2)-pow(var_ratio,-2)));
        const double dr = dist*2*(var_ratio-pow(var_ratio,-3));
        dvariance  [i]+= dr/mean_var;
        const double drn=
          dist*(mean_var/pow(variance[i],2)-pow(var_ratio,2)/mean_var);
        dvariance  [(ss+0)*nfs+(fs-1)]+=drn;
        dvariance  [(ss+0)*nfs+(fs+1)]+=drn;
        // smoothness of background
        const double l = 0.5*dist*pow(mean_val-background[i],2)
                            /(mean_var+variance[i]);
        value += l;
        dbackground[i]+=dist*(background[i]-mean_val)/(mean_var+variance[i]);
        dvariance  [i]+=-0.5*dist*
          pow((mean_val-background[ss*nfs+fs])/(mean_var+variance[i]),2);
        const double dmean_val_n = 0.5*dist*(mean_val-background[i])
                                           /(mean_var+variance[i]); 
        const double dmean_var_n = -0.25*dist*
          pow((mean_val-background[i])/(mean_var+variance[i]),2);
        dbackground[(ss+0)*nfs+(fs-1)]+=dmean_val_n;
        dbackground[(ss+0)*nfs+(fs+1)]+=dmean_val_n;
        dvariance  [(ss+0)*nfs+(fs-1)]+=dmean_var_n;
        dvariance  [(ss+0)*nfs+(fs+1)]+=dmean_var_n;
      }
      if ((ss-1<nss)&&(ss+1<nss)) {
        const double mean_val =
          0.5*(background[(ss-1)*nfs+(fs+0)]+background[(ss+1)*nfs+(fs+0)]);
        const double mean_var =
          0.5*(  variance[(ss-1)*nfs+(fs+0)]+  variance[(ss+1)*nfs+(fs+0)]);
        // smoothness of variance
        const double var_ratio = variance[i]/mean_var;
        value+=dist*(log(sqrt(pi)/2)-(2-pow(var_ratio,2)-pow(var_ratio,-2)));
        const double dr = dist*2*(var_ratio-pow(var_ratio,-3));
        dvariance  [i]+= dr/mean_var;
        const double drn=
          dist*(mean_var/pow(variance[i],2)-pow(var_ratio,2)/mean_var);
        dvariance  [(ss-1)*nfs+(fs+0)]+=drn;
        dvariance  [(ss+1)*nfs+(fs+0)]+=drn;
        // smoothness of background
        const double l = 0.5*dist*pow(mean_val-background[i],2) 
                            /(mean_var+variance[i]);
        value += l;
        dbackground[i]+=dist*(background[i]-mean_val)/(mean_var+variance[i]);
        dvariance  [i]+=-0.5*
          dist*pow((mean_val-background[ss*nfs+fs])/(mean_var+variance[i]),2);
        const double dmean_val_n = 0.5*dist*(mean_val-background[i])
                                           /(mean_var+variance[i]); 
        const double dmean_var_n = -0.25*dist*
          pow((mean_val-background[i])/(mean_var+variance[i]),2);
        dbackground[(ss-1)*nfs+(fs+0)]+=dmean_val_n;
        dbackground[(ss+1)*nfs+(fs+0)]+=dmean_val_n;
        dvariance  [(ss-1)*nfs+(fs+0)]+=dmean_var_n;
        dvariance  [(ss+1)*nfs+(fs+0)]+=dmean_var_n;
      }
    }
  }
  return value;
}

double background_target_(
    const size_t nfs,
    const size_t nss,
    const double a,
    const double dist,
    const double* image,
    const double* background,
    const double* variance,
          double* dbackground,
          double* dvariance
    ){
  for (size_t j=0;j!=nfs*nss;++j) dbackground[j]=0;
  for (size_t j=0;j!=nfs*nss;++j) dvariance  [j]=0;
  double value = 0;
  for (size_t ss=0;ss!=nss;++ss) {
    for (size_t fs=0;fs!=nfs;++fs) {
      const size_t i = ss*nfs+fs;
      if (true) { // is not masked, there is no mask here
        const auto tmp =
          llg(image[i],background[i],variance[i],a,universal_distribution);
        value += get<0>(tmp);
        dbackground[ss*nfs+fs] += get<0>(get<1>(tmp));
          dvariance[ss*nfs+fs] += get<1>(get<1>(tmp));
      }
      double mean_val = 0;
      double mean_var = 0;
      size_t n = 0;
      if ((fs-1<nfs)&&(fs+1<nfs)) {
        mean_val+=
          2*(background[(ss+0)*nfs+(fs-1)]+background[(ss+0)*nfs+(fs+1)]);
        mean_var+=
          2*(  variance[(ss+0)*nfs+(fs-1)]+  variance[(ss+0)*nfs+(fs+1)]);
        n+=4;
      }
      if ((ss-1<nss)&&(ss+1<nss)) {
        mean_val+=
          2*(background[(ss-1)*nfs+(fs+0)]+background[(ss+1)*nfs+(fs+0)]);
        mean_var+=
          2*(  variance[(ss-1)*nfs+(fs+0)]+  variance[(ss+1)*nfs+(fs+0)]);
        n+=4;
      }
      if ((fs-1<nfs)&&(fs+1<nfs)&&(ss-1<nss)&&(ss+1<nss)) {
        mean_val+=background[(ss-1)*nfs+(fs-1)]+background[(ss+1)*nfs+(fs+1)];
        mean_var+=  variance[(ss-1)*nfs+(fs-1)]+  variance[(ss+1)*nfs+(fs+1)];
        n+=2;
      }
      if ((fs-1<nfs)&&(fs+1<nfs)&&(ss-1<nss)&&(ss+1<nss)) {
        mean_val+=background[(ss+1)*nfs+(fs-1)]+background[(ss-1)*nfs+(fs+1)];
        mean_var+=  variance[(ss+1)*nfs+(fs-1)]+  variance[(ss-1)*nfs+(fs+1)];
        n+=2;
      }
      if (n==0) continue;
      mean_val/=n;
      mean_var/=n;
      const double var_ratio = variance[i]/mean_var;
      // smoothness of variance
      value+=dist*(log(sqrt(pi)/2)-(2-pow(var_ratio,2)-pow(var_ratio,-2)));
      const double dr = dist*2*(var_ratio-pow(var_ratio,-3));
      dvariance  [i]+= dr/mean_var;
      const double drn=
        dist*2*(mean_var/pow(variance[i],2)-pow(var_ratio,2)/mean_var)/n;
      mean_var+=variance[i];
      // smoothness of background
      const auto [p_g,dg] = gauss(mean_val,background[ss*nfs+fs],mean_var);
      value+=-dist*log(p_g);
      dbackground[i]+=-dist*get<1>(dg)/p_g;
      dvariance  [i]+=-dist*get<2>(dg)/p_g;
      const double dmean_val_n = -dist*get<0>(dg)/(n*p_g);
      const double dmean_var_n = -dist*get<2>(dg)/(n*p_g);
      if ((fs-1<nfs)&&(fs+1<nfs)) {
        dbackground[(ss+0)*nfs+(fs-1)]+=2.0*dmean_val_n;
        dbackground[(ss+0)*nfs+(fs+1)]+=2.0*dmean_val_n;
        dvariance  [(ss+0)*nfs+(fs-1)]+=2.0*dmean_var_n;
        dvariance  [(ss+0)*nfs+(fs+1)]+=2.0*dmean_var_n;
        dvariance  [(ss+0)*nfs+(fs-1)]+=2.0*drn;
        dvariance  [(ss+0)*nfs+(fs+1)]+=2.0*drn;
      }
      if ((ss-1<nss)&&(ss+1<nss)) {
        dbackground[(ss-1)*nfs+(fs+0)]+=2.0*dmean_val_n;
        dbackground[(ss+1)*nfs+(fs+0)]+=2.0*dmean_val_n;
        dvariance  [(ss-1)*nfs+(fs+0)]+=2.0*dmean_var_n;
        dvariance  [(ss+1)*nfs+(fs+0)]+=2.0*dmean_var_n;
        dvariance  [(ss-1)*nfs+(fs+0)]+=2.0*drn;
        dvariance  [(ss+1)*nfs+(fs+0)]+=2.0*drn;
      }
      if ((fs-1<nfs)&&(fs+1<nfs)&&(ss-1<nss)&&(ss+1<nss)) {
        dbackground[(ss-1)*nfs+(fs-1)]+=1.0*dmean_val_n;
        dbackground[(ss+1)*nfs+(fs+1)]+=1.0*dmean_val_n;
        dvariance  [(ss-1)*nfs+(fs-1)]+=1.0*dmean_var_n;
        dvariance  [(ss+1)*nfs+(fs+1)]+=1.0*dmean_var_n;
        dvariance  [(ss-1)*nfs+(fs-1)]+=1.0*drn;
        dvariance  [(ss+1)*nfs+(fs+1)]+=1.0*drn;
      }
      if ((fs-1<nfs)&&(fs+1<nfs)&&(ss-1<nss)&&(ss+1<nss)) {
        dbackground[(ss+1)*nfs+(fs-1)]+=1.0*dmean_val_n;
        dbackground[(ss-1)*nfs+(fs+1)]+=1.0*dmean_val_n;
        dvariance  [(ss+1)*nfs+(fs-1)]+=1.0*dmean_var_n;
        dvariance  [(ss-1)*nfs+(fs+1)]+=1.0*dmean_var_n;
        dvariance  [(ss+1)*nfs+(fs-1)]+=1.0*drn;
        dvariance  [(ss-1)*nfs+(fs+1)]+=1.0*drn;
      }
    }
  }
  return value;
}

void test_derivative(
    const size_t nfs,
    const size_t nss,
    const double a,
    const double dist,
    const double* image,
    double* background,
    double* variance
    ) {
  double eps = 0.001;
  double* dbackground = new double[nfs*nss];
  double* dvariance   = new double[nfs*nss];
  for (size_t i=0;i!=nfs*nss;++i) {
    const double b = background[i];
    background[i]=b+eps;
    const double pb = background_target
      (nfs,nss,a,dist,image,background,variance,dbackground,dvariance);
    background[i]=b-eps;
    const double mb = background_target
      (nfs,nss,a,dist,image,background,variance,dbackground,dvariance);
    background[i]=b;
    const double v = variance[i];
    variance[i]=v+eps;
    const double pv = background_target
      (nfs,nss,a,dist,image,background,variance,dbackground,dvariance);
    variance[i]=v-eps;
    const double mv = background_target
      (nfs,nss,a,dist,image,background,variance,dbackground,dvariance);
    variance[i]=v;
    cout << dbackground[i] << " " << (pb-mb)/(2*eps) << " "
         << dvariance[i] << " " << (pv-mv)/(2*eps) << endl;
  }
}

void optimise_background(
    const size_t nfs,
    const size_t nss,
    const double a,
    const double dist,
    const double* image,
    double* background,
    double* variance
    ) {
  const double  phi = (sqrt(5.0) + 1.0) * 0.5;
  const double iphi = 1.0/phi;
  double beta0 = 1;
  double beta1 = 1;
  double* dbackground = new double[nfs*nss];
  double* dvariance   = new double[nfs*nss];
  double* background_ = new double[nfs*nss];
  double* variance_   = new double[nfs*nss];
  double* dump0       = new double[nfs*nss];
  double* dump1       = new double[nfs*nss];
  for (size_t i=0;i!=(1u<<12);++i) {
    //cerr << '\r' << setw(8) << i;
    const double value = background_target
      (nfs,nss,a,dist,image,background,variance,dbackground,dvariance);
    for (size_t i=0;i!=nss*nfs;++i) {
      //cout << dbackground[i] << " " << dvariance[i] << endl;
    }
    cerr << setw(8) << i << " " << value << endl;
    if (i%2==0) {
      double norm = 0;
      for (size_t i=0;i!=nfs*nss;++i) norm+=pow(dbackground[i],2);
      norm = sqrt(norm)/(nfs*nss);
      while(beta0>1e-32) {
        for (size_t i=0;i!=nfs*nss;++i)
          background_[i]=background[i]-beta0/norm*dbackground[i];
        const double test = background_target
          (nfs,nss,a,dist,image,background_,variance,dump0,dump1);
        //cerr << beta0 << " " << test << endl;
        if (test<value) {
          beta0*=2;
          swap(background_,background);
          break;
        }
        beta0*=iphi;
        continue;
      }
      if (beta0>1e-32) continue;
      break;
    } else {
      double norm = 0;
      for (size_t i=0;i!=nfs*nss;++i) norm+=pow(dvariance[i],2);
      norm = sqrt(norm)/(nfs*nss);
      while(beta1>1e-32) {
        for (size_t i=0;i!=nfs*nss;++i) 
          variance_[i]=abs(variance[i]-beta1/norm*dvariance[i]);
        const double test = background_target
          (nfs,nss,a,dist,image,background,variance_,dump0,dump1);
        //cerr << beta1 << " " << test << endl;
        if (test<value) {
          beta1*=2;
          swap(variance_,variance);
          break;
        }
        beta1*=iphi;
        continue;
      }
      if (beta1>1e-32) continue;
      break;
    }
  }
}

void optimise_background2(
    const size_t nfs,
    const size_t nss,
    const double a,
    const double dist,
    const double* image,
    double* background,
    double* variance
    ) {
  double beta = 1.0/256;
  double* dbackground = new double[nfs*nss];
  double* dvariance   = new double[nfs*nss];
  for (size_t i=0;i!=(1u<<12);++i) {
    //cerr << '\r' << setw(8) << i;
    const double value = background_target
      (nfs,nss,a,dist,image,background,variance,dbackground,dvariance);
    cerr << setw(8) << i << " " << value << endl;
    double norm_dbackground = 0;
    for (size_t i=0;i!=nfs*nss;++i) norm_dbackground+=pow(dbackground[i],2);
    norm_dbackground = beta/sqrt(norm_dbackground)*(nfs*nss);
    double norm_dvariance = 0;
    for (size_t i=0;i!=nfs*nss;++i) norm_dvariance+=pow(dvariance[i],2);
    norm_dvariance = beta/sqrt(norm_dvariance)*(nfs*nss);
    for (size_t i=0;i!=nfs*nss;++i)
      background[i]-=norm_dbackground*dbackground[i];
    for (size_t i=0;i!=nfs*nss;++i)
      variance[i]=abs(variance[i]-norm_dvariance*dvariance[i]);
  }
}

int main() {
  const double a = 1.0/256;
  const auto g = 9511.11*0.00105;
  const size_t nfs = 256;
  const size_t nss = 256;
  double* data = new double[nfs*nss];
  std::random_device gen;
  ofstream nonoise("test_background_nonoise.pgm");
  ofstream noise  ("test_background.pgm");
  noise   << "P5 256 256 255" << endl;
  nonoise << "P5 256 256 255" << endl;
  double* background = new double[nfs*nss];
  double* variance = new double[nfs*nss];
  for (size_t ss=0;ss!=nss;++ss) {
    for (size_t fs=0;fs!=nfs;++fs) {
      double x = ss-128.0;
      double y = fs-128.0;
      const auto tmp = true_background(x,y,gen,a,g);
      data[ss*nfs+fs] = get<1>(tmp);
      nonoise.put(uint8_t(clip(get<0>(tmp),0,255)));
      noise.put  (uint8_t(clip(get<1>(tmp),0,255)));
      background[ss*nfs+fs]=0;
      variance[ss*nfs+fs]=256;
    }
  }
  //test_derivative(nfs,nss,a,1,data,background,variance); return 0;
  optimise_background2(nfs,nss,0.5,256,data,background,variance);
  optimise_background(nfs,nss,0.5,256,data,background,variance);
  ofstream smooth("test_background_smooth.pgm");
  smooth << "P5 256 256 255" << endl;
  for (size_t i=0;i!=nss*nfs;++i)
    smooth.put(uint8_t(clip(background[i],0,255)));
  ofstream smooth_variance("test_background_variance.pgm");
  smooth_variance << "P5 256 256 255" << endl;
  for (size_t i=0;i!=nss*nfs;++i)
    smooth_variance.put(uint8_t(clip(variance[i]/g,0,255)));
}
