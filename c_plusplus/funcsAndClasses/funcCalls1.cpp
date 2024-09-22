// Outputs three random numbers.

#include <iostream>     // Declaration of cin and cout
#include <cmath>        // Prototypes of srand() and rand()
                        // void srand(unsigned int seed);
                        // int rand(void);

using namespace std;

int main()
{
    unsigned int seed;
    int z1, z2, z3;

    cout << "Enter a seed value: ";
    cin >> seed;
    cout <<endl;

    srand(seed); // When an integer seed number, use it as an argument for new
                    // sequence of random numbers

    z1 = rand(); // compute three random numbers
    z2 = rand();
    z3 = rand();
}