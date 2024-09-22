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

    cout << "2.5 raised to the power 3 is " << y << endl;

    cout << y << endl;

    // Calculating with pow() is not always the best solution but is possible
    cout << "5 raised to power of 2.5 is " << pow(5.0, x) << endl;
    
    return 0;
}