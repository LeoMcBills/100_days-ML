// Write an exchange rate program

#include <iostream>
#include <iomanip>
using namespace std;

int main()
{
    double rate = 1.15;

    cout << fixed << setprecision(2);
    cout << "\tEuro" << "\tDollar" << endl;
    
    for (int euro = 1; euro <= 5; ++euro)
    {
        double dollar = euro * rate;
        cout << "\t" << euro << "\t" << dollar << endl;
    }

    return 0;
}