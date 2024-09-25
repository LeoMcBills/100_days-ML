// defining functions
#include <iostream>
using namespace std;

void test( int, double );

int main()
{
    cout << "\nNow function test() will be called.\n";
    test( 10, -7.5);
    cout << "\nAnd back again in main()." << endl;

    return 0;
}

void test(int arg1, double arg2)
{
    cout << "\nIn function test()."
            << "\n 1. argument: " << arg1
            << "\n 2. argument: " << arg2 << endl;
}