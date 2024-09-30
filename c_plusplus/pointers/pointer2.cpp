// Definition and call of function swap().
// Demonstrates the use of pointers as parameters.

#include <iostream>
using namespace std;

void swap ( float *, float *);

int main()
{
    float x = 11.1F;
    float y = 22.2F;

    cout << "Formerly x was " << x << " and y was " << y << endl;

    swap( &x, &y);

    cout << "x is now " << x << " and y is now " << y << endl;
    return 0;
}

void swap( float *p1, float *p2)
{
    float temp;

    temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}