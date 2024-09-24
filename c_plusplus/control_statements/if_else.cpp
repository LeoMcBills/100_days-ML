// Demonstrates the use of if-else statements

#include <iostream>
using namespace std;

int main()
{
    float x, y, min;

    cout << "Enter two different numbers:\n";
    
    if (cout << "-------"; cin >> x && cin >> y)
    {
        if (x > y){
            min = y;
        } else {
            min = x;
        }
        cout << "\nThe smaller number is: " << min << endl;
    } else {
        cout << "\nInvalid input!" << endl;
    }

    return 0;
}