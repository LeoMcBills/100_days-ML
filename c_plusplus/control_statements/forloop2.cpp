// Output a table of exchange:
// Euro and US-$

#include <iostream>
#include <iomanip>
using namespace std;

int main()
{
    long euro, maxEuro, lower, upper, step;
    double rate;

    cout << "\n***Table of exchange "<< " Euro - US-$***\n\n"
            << "\nPlease give the rate of exchange: " 
            << " one Euro in US-$: ";
        
    cin >> rate;

    cout << "\nPlease enter the maximum euro: ";
    cin >> maxEuro;

    // -----------------------Output Table----------------------------------
    cout << "\n"
            << setw(12) << "Euro"
            << setw(20) << "US-$"
            << "\t\tRate: " << rate << endl;
    
    cout << fixed << setprecision(2) << endl;

    for (lower = 1, step = 1; lower <= maxEuro; step *= 10, lower = 2 * step)
    {
        for (euro = lower, upper = step*10; euro <= upper && euro <= maxEuro; euro += step)
        {
            cout << setw(12) << euro << setw(20) << euro * rate << endl;
        }
    }

    return 0;

}