// Example for a simple function returning a value.
#include <iostream>
#include <iomanip>
using namespace std;

double area(double, double);

int main()
{
    double x = 3.5, y = 7.2, res;

    res = area( x, y+1 );

    // To output to two decimal places:
    cout << fixed << setprecision(2);
    cout << "\nThe area of a rectangle "
            << "\nwith wwidth   " << setw(5) << x
            << "\n and length   " << setw(5) << y+1
            << "\n is           " << setw(5) << res
            << endl;
    
    return 0;
}

// Defining the function area():
// Computes the area of a rectangle.
double area (double width, double len)
{
    return (width * len);
}