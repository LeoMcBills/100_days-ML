// Demonstrates the definition and use of references.
// -------------------------------------------------------------------
#include <iostream>
#include <string>
using namespace std;

float x = 10.7F;                                 // Global

int main()
{
    float &rx = x;          // Local reference to x

    rx *= 2;

    cout << "   x = " << x << endl
         << "   rx = "<< rx<< endl;

    const float& cref = x;
    cout << "cref = " << cref << endl;

    const string str = "I am a constant string!";
    const string& text = str;
    cout << text << endl;

    return 0; 
}