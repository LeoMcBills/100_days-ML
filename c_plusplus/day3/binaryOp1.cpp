#include <iostream>
using namespace std;

int main()
{
    double x, y;

    cout << "\nEnter two floating point values: ";

    cin >> x >> y;

    cout << "The average of the two numbers is: "
        << (x + y) / 2.0 << endl;

    return 0;
}