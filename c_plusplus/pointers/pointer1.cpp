// Prints the values and addresses of variables.
#include <iostream>
using namespace std;

int var, *ptr;      

int main()
{
    var = 100;
    ptr = &var;

    cout << " Value of var:         " << var
         << "\n Address of var:       " << &var
         << endl;

    cout << " Value of ptr: "         << ptr
         << "\n Address of ptr: "       << &ptr
         << endl;

    return 0;
}