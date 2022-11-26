#include <iostream>

using std::cin;
using std::cout;

int main() {
  for (int16_t n;cin.read(reinterpret_cast<char*>(&n),2);) {
    if (!cin) break;
    uint8_t m = n/256;
    cout.write(reinterpret_cast<char*>(&m),1);
  }
}
