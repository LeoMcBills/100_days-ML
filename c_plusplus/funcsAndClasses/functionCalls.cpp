// Calculating powers with the standard function pow()

#include <iostream>
#include <cmath>

using namespace std;

int main() {
    double x = 2.5, y;

    // By means of a prototype, the compiler generates the correct
    // function call or an error message!
    // Computes x raised to the power 3:

    y = pow(x, 3.0);

    cout << y << endl;
    
    return 0;
}