#include <iostream>
using namespace std;

int main()
{
    float x, y;

    cout << "Type two different numbers: \n";

    if (!(cout << "==> ", cin >> x && cout << "==> ", cin >> y))
    {
        cout << "\nInvalid input!" << endl;
    }
    else {
        cout << "\nThe greater value is: "
                << (x > y? x : y)
                << endl;
    }

    return 0;
}