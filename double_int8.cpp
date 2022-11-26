#include <iostream>

using std::cin;
using std::cout;

int main() {
  for (double n;cin.read(reinterpret_cast<char*>(&n),sizeof(double));) {
    uint8_t m = n/256;
    cout.write(reinterpret_cast<char*>(&m),1);
  }
}
