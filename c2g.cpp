#include <iostream>
#include <iomanip>
#include "geometry.hpp"

using std::cin;
using std::cout;
using std::endl;

int main(){
  double coffset;
  cin >> coffset;
  auto f=geometry::get_crystfel_geometry(cin,coffset);
  for (auto it=f.transforms.begin();it!=f.transforms.end();++it){
    cout << it->max_fs-it->min_fs+1 << " "
         << it->max_ss-it->min_ss+1 << endl;
    cout << it->fs_2_x << " " << it->ss_2_x << " " << it->xoffset << endl; 
    cout << it->fs_2_y << " " << it->ss_2_y << " " << it->yoffset << endl; 
    cout << it->fs_2_z << " " << it->ss_2_z << " " << it->zoffset << endl; 
  }
}

