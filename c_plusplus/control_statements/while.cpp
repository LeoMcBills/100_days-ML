// A demonstration of a while loop
// This program calculates average of numbers

#include <iostream>
using namespace std;

int main()
{
    int x, count(0);
    float sum(0.0);

    cout << "Enter numbers whose average is to be calculated: ";

    while (cin >> x)
    {
        sum += x;
        ++count;
        cout << "The average of the numbers entered so far is " << sum / count << endl;
    }    

    return 0;
}